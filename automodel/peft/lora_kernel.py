import triton
import triton.language as tl

import torch

assert triton.runtime.driver.active.get_current_target().backend == "cuda"


def addmm_kernel_wrapper(a, b, res, scale, inplace_add=False, dtype=torch.float32):
    assert a.shape[1] == b.shape[0], "Incompatible A and B dimensions"
    # assert a.is_contiguous(), "Matrix a must be contiguous"
    if res is not None:
        assert a.shape[0] == res.shape[0], "Incompatible A and C dimensions"
        assert b.shape[1] == res.shape[1], "Incompatible B and C dimensions"

    M, K = a.shape
    K, N = b.shape

    BLOCK_M = 32
    BLOCK_N = 16
    BLOCK_K = 256 if M < 128 else 32
    GROUP_M = 8

    if res is None:
        res = torch.empty((M, N), device=a.device, dtype=dtype)

    grid = (triton.cdiv(M, 32) * triton.cdiv(N, 16),)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    matmul_kernel[grid](
        a,
        b,
        res,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        res.stride(0),
        res.stride(1),
        scale,
        inplace_add,  #
        # BLOCK_M,
        # BLOCK_N,
        # BLOCK_K,
        # GROUP_M,
    )

    return res


def get_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        # Good config for fp8 inputs.
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    # scale factor
    scale,
    # True if c += a @ b, False if c is overwritten by a @ b
    inplace_add,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
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

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # TODO: add properly
    res = scale * accumulator
    res = res.to(a_ptr.dtype.element_ty)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if inplace_add:
        orig = tl.load(c_ptrs, mask=c_mask)
        res = res + orig
    tl.store(c_ptrs, res, mask=c_mask)



@triton.jit
def tri_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    M,
    N,
    K,
    L,
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cn,
    stride_cl,
    stride_dm,
    stride_dl,
    # scale factor
    scale,
    # True if d += a @ b @ c, False if d is overwritten
    inplace_add,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    BLOCK_SIZE_L: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the matmul D = A x B x C.
    A has shape (M, K), B has shape (K, N), C has shape (N, L), and D has shape (M, L)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_l = tl.cdiv(L, BLOCK_SIZE_L)
    num_pid_in_group = GROUP_SIZE_M * num_pid_l
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_l = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_cl = (pid_l * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)) % L
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    c_ptrs = c_ptr + (offs_n[:, None] * stride_cn + offs_cl[None, :] * stride_cl)


    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_L), dtype=tl.float32)
    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn) + n * BLOCK_SIZE_N * stride_bn
        # -----------------------------------------------------------
        # Iterate to compute a block of the AB matrix.
        # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        # of fp32 values for higher accuracy.
        ab = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            # We accumulate along the K dimension.
            ab += tl.dot(a, b, out_dtype=tl.float32)
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        c = tl.load(c_ptrs, mask=offs_n[:, None] < N - n * BLOCK_SIZE_N, other=0.0)
        accumulator += tl.dot(ab, c.to(tl.float32), out_dtype=tl.float32)
        c_ptrs += BLOCK_SIZE_N * stride_cn

    res = scale * accumulator
    res = res.to(a_ptr.dtype.element_ty)

    # -----------------------------------------------------------
    # Write back the block of the output matrix D with masks.
    offs_dm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_dl = pid_l * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    d_ptrs = d_ptr + stride_dm * offs_dm[:, None] + stride_dl * offs_dl[None, :]
    d_mask = (offs_dm[:, None] < M) & (offs_dl[None, :] < L)
    if inplace_add:
        orig = tl.load(d_ptrs, mask=d_mask)
        res = res + orig
    tl.store(d_ptrs, res, mask=d_mask)




def tri_matmul_wrapper(a, b, c, res, scale, inplace_add=False, dtype=torch.float32):
    assert a.shape[1] == b.shape[0], "Incompatible A and B dimensions"
    assert b.shape[1] == c.shape[0], "Incompatible B and C dimensions"
    if res is not None:
        assert a.shape[0] == res.shape[0], "Incompatible A and D dimensions"
        assert c.shape[1] == res.shape[1], "Incompatible C and D dimensions"

    M, K = a.shape
    K, N = b.shape
    N, L = c.shape

    BLOCK_M = 32
    BLOCK_N = 16
    BLOCK_K = 256 if M < 128 else 32
    BLOCK_L = 512
    GROUP_M = 8

    if res is None:
        res = torch.empty((M, L), device=a.device, dtype=dtype)

    grid = (triton.cdiv(M, 32) * triton.cdiv(L, 16),)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(L, META['BLOCK_SIZE_L']),)

    tri_matmul_kernel[grid](
        a,
        b,
        c,
        res,  #
        M,
        N,
        K,
        L, #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1), #
        res.stride(0),
        res.stride(1),
        scale,
        inplace_add,  #
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        BLOCK_L,
        GROUP_M,
    )

    return res

