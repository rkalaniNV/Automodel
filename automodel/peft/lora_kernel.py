# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import triton
import triton.language as tl

import torch


@triton.jit
def inner_kernel_test(a_ptr, b_ptr, c_ptr,
                    M, K, N,
                    stride_am, stride_ak,
                    stride_bk, stride_bn,
                    stride_cm, stride_cn,
                    BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr,
                    GROUP_SIZE_M: tl.constexpr,
                    scale):

    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    ab = inner_kernel(pid_m, pid_n,
                    a_ptr, b_ptr,
                    M, K, N,
                    stride_am, stride_ak,
                    stride_bk, stride_bn,
                    BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N,
                    scale)
    
    offs_cm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_cn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, ab, mask=c_mask)


@triton.jit
def inner_kernel(pid_m, pid_n,
                 a_ptr, b_ptr,
                 M, K, N,
                 stride_am, stride_ak,
                 stride_bk, stride_bn,
                 BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N,
                 scale):

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    ab = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        b_mask = (offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        ab += tl.dot(a, b, out_dtype=tl.float32)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    return scale * ab


@triton.jit
def mat_vec_mul(pid_m, pid_n,
                ab_result, c_ptr, d_ptr,
                M, N, L,
                stride_cn, stride_cl, stride_dm, stride_dl,
                BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_L):
    offs_cn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_l = tl.arange(0, BLOCK_SIZE_L)
    offs_dm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))

    c_ptrs = c_ptr + (offs_cn[:, None] * stride_cn + offs_l[None, :] * stride_cl) 

    d_ptrs = d_ptr + stride_dm * offs_dm[:, None] + stride_dl * offs_l[None, :]
    d_mask = (offs_dm[:, None] < M) & (offs_l[None, :] < L)
    c_mask = (offs_cn[:, None] < N) & (offs_l[None, :] < L)

    for l in tl.range(0, tl.cdiv(L, BLOCK_SIZE_L)):
        d_mask = (offs_dm[:, None] < M) & (offs_l[None, :] < L - l * BLOCK_SIZE_L)
        c_mask = (offs_cn[:, None] < N) & (offs_l[None, :] < L - l * BLOCK_SIZE_L)
        c = tl.load(c_ptrs, mask=c_mask, other=0.0)

        abc = tl.dot(ab_result, c)
        # TODO: make sure this doesn't break autotune?
        orig = tl.load(d_ptrs, mask=d_mask, other=0.0)
        tl.store(d_ptrs, abc + orig, mask=d_mask)
        
        c_ptrs += BLOCK_SIZE_L * stride_cl
        d_ptrs += BLOCK_SIZE_L * stride_dl


@triton.heuristics(values={'BLOCK_SIZE_N': lambda args: max(triton.next_power_of_2(args['N']), 16)})
@triton.jit
def lora_forward_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    M, N, K, L,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cn, stride_cl,
    stride_dm, stride_dl,
    # scale factor
    scale,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    BLOCK_SIZE_L: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,  #
):

    """Kernel for computing the matmul D = A x B x C.
    A has shape (M, K), B has shape (K, N), C has shape (N, L), and D has shape (M, L)
    N, the LoRA dimension must be less than or equal to than BLOCK_SIZE_N.
    """
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m


    ab_result = inner_kernel(pid_m, pid_n,
                 a_ptr, b_ptr,
                 M, K, N,
                 stride_am, stride_ak,
                 stride_bk, stride_bn,
                 BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N,
                 scale)
    # -----------------------------------------------------------
    ab_result = ab_result.to(c_ptr.dtype.element_ty)

    mat_vec_mul(pid_m, pid_n,
                ab_result, c_ptr, d_ptr,
                M, N, L,
                stride_cn, stride_cl, stride_dm, stride_dl,
                BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_L)



def lora_forward_wrapper(a, b, c, res, scale, dtype=torch.float32):
    assert a.shape[1] == b.shape[0], "Incompatible A and B dimensions"
    assert b.shape[1] == c.shape[0], "Incompatible B and C dimensions"
    if res is not None:
        assert a.shape[0] == res.shape[0], "Incompatible A and D dimensions"
        assert c.shape[1] == res.shape[1], "Incompatible C and D dimensions"

    M, K = a.shape
    K, N = b.shape
    N, L = c.shape

    BLOCK_M = 32
    BLOCK_K = 256
    BLOCK_L = 256
    GROUP_M = 8
    
    if res is None:
        res = torch.zeros((M, L), device=a.device, dtype=dtype)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    lora_forward_kernel[grid](
        a, b, c, res,
        M, N, K, L, #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1), 
        c.stride(0), c.stride(1), #
        res.stride(0), res.stride(1),
        scale,
        BLOCK_M,
        BLOCK_K,
        BLOCK_L,
        GROUP_M,
    )

    return res


#======================
def lora_update_wrapper(a, b, c, scale, dtype=torch.float32):
    assert a.shape[1] == b.shape[0], "Incompatible A and B dimensions"
    assert b.shape[1] == c.shape[0], "Incompatible B and C dimensions"

    M, K = a.shape
    K, N = b.shape
    N, L = c.shape

    BLOCK_M = 32
    BLOCK_K = 256
    BLOCK_L = 256
    GROUP_M = 8
    
    res = torch.zeros((M, L), device=a.device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    lora_forward_kernel[grid](
        a, b, c, res,
        M, N, K, L, #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1), 
        c.stride(0), c.stride(1), #
        res.stride(0), res.stride(1),
        scale,
        BLOCK_M,
        BLOCK_K,
        BLOCK_L,
        GROUP_M,
    )

    return res


def inner_kernel_wrapper(a, b, scale, dtype=torch.float32):
    assert a.shape[1] == b.shape[0], "Incompatible A and B dimensions"

    M, K = a.shape
    K, N = b.shape
    BLOCK_M = 32
    BLOCK_N = 16
    BLOCK_K = 256
    BLOCK_L = 256
    GROUP_M = 8
    
    c = torch.zeros((M, N), device=a.device, dtype=dtype)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    inner_kernel_test[grid](a, b, c,
                    M, K, N,
                    a.stride(0), a.stride(1),  #
                    b.stride(0), b.stride(1), 
                    c.stride(0), c.stride(1), #
                    BLOCK_M, BLOCK_K, BLOCK_N, GROUP_M,
                    scale)

    return c