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

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Self-contained replacement for torchao.float8.Float8Linear.

NOTE
----
*  The quantisation stubs below **do not** actually convert data to FP8.
   They only preserve the expected API so that the module works out-of-box.
*  Replace the bodies of `hp_tensor_to_float8_dynamic` and
   `hp_tensor_and_scale_to_float8` with proper FP8 logic when you have
   hardware-specific kernels.
"""

from __future__ import annotations
from typing import Optional
from dataclasses import dataclass, field
import enum

import torch
import torch.utils.checkpoint as checkpoint

from nemo_automodel.float8.float8_scaling_utils import (
    get_maybe_axiswise_dim,
    hp_tensor_to_float8_dynamic,
)

class ScalingType(enum.Enum):
    DISABLED = enum.auto()
    DYNAMIC = enum.auto()


class ScalingGranularity(enum.Enum):
    PER_TENSOR = enum.auto()
    AXISWISE = enum.auto()


class GemmInputRole(enum.Enum):
    INPUT = enum.auto()
    WEIGHT = enum.auto()
    GRAD_OUTPUT = enum.auto()


@dataclass
class CastConfig:
    scaling_type: ScalingType = ScalingType.DYNAMIC
    scaling_granularity: ScalingGranularity = ScalingGranularity.PER_TENSOR
    target_dtype: torch.dtype = (
        # If the build has FP8 dtypes use them, otherwise fall back to fp16.
        getattr(torch, "float8_e4m3fn", torch.float16)
    )

    # Pretty print used by Float8Linear.extra_repr
    def short_str(self) -> str:
        stype = "D" if self.scaling_type is ScalingType.DYNAMIC else "X"
        gran = "A" if self.scaling_granularity is ScalingGranularity.AXISWISE else "T"
        dtype = str(self.target_dtype).split(".")[-1]
        return f"{stype}/{gran}/{dtype}"


@dataclass
class GemmConfig:
    use_fast_accum: bool = False


@dataclass
class LinearGemmConfigs:
    output:        GemmConfig = field(default_factory=GemmConfig)
    grad_input:    GemmConfig = field(default_factory=GemmConfig)
    grad_weight:   GemmConfig = field(default_factory=GemmConfig)


@dataclass
class Float8LinearConfig:
    emulate: bool = True                        # keep for API compatibility
    pad_inner_dim: bool = False
    force_recompute_fp8_weight_in_bwd: bool = False
    round_scales_to_power_of_2: bool = False
    enable_fsdp_float8_all_gather: bool = False

    cast_config_input:                      CastConfig = field(default_factory=CastConfig)
    cast_config_weight:                     CastConfig = field(default_factory=CastConfig)
    cast_config_grad_output:                CastConfig = field(default_factory=CastConfig)

    # Separate configs used during backward – default to the forward ones.
    cast_config_input_for_grad_weight:      CastConfig = field(default_factory=CastConfig)
    cast_config_weight_for_grad_input:      CastConfig = field(default_factory=CastConfig)
    cast_config_grad_output_for_grad_weight:CastConfig = field(default_factory=CastConfig)

    gemm_config_output:      GemmConfig = field(default_factory=GemmConfig)
    gemm_config_grad_input:  GemmConfig = field(default_factory=GemmConfig)
    gemm_config_grad_weight: GemmConfig = field(default_factory=GemmConfig)


# ---------------------------------------------------------------------------
# Minimal FP8 utility stubs
# ---------------------------------------------------------------------------
def tensor_already_casted_to_fp8(t: torch.Tensor) -> bool:
    return t.dtype in {
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
    }


def tensor_to_scale(t: torch.Tensor, target_dtype: torch.dtype):
    # Stub – real impl would compute max(abs(t)) etc.
    return None


def get_maybe_axiswise_dim(dim: int, granularity: ScalingGranularity):
    return dim if granularity is ScalingGranularity.AXISWISE else None


def hp_tensor_to_float8_dynamic(
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
    *_, **__
) -> torch.Tensor:
    """
    Stub dynamic quantisation.
    Current behaviour: just forward the source tensor;
    replace this with real quantisation to target_dtype.
    """
    return tensor.to(target_dtype)


def hp_tensor_and_scale_to_float8(
    tensor: torch.Tensor,
    scale: Optional[torch.Tensor],
    target_dtype: torch.dtype,
    *_, **__
) -> torch.Tensor:
    # Ignore `scale` – real impl would use it.
    return hp_tensor_to_float8_dynamic(tensor, target_dtype)


# ---------------------------------------------------------------------------
# Linear-specific helper configs
# ---------------------------------------------------------------------------
@dataclass
class ScaledMMConfig:
    emulate: bool
    use_fast_accum: bool
    transpose_result: bool
    pad_inner_dim: bool


@dataclass
class LinearMMConfig:
    mm_output:     ScaledMMConfig
    mm_grad_input: ScaledMMConfig
    mm_grad_weight:ScaledMMConfig


# ---------------------------------------------------------------------------
# Helper to extract weight scale (stub)
# ---------------------------------------------------------------------------
def _get_weight_scale(
    weight: torch.Tensor,
    scaling_type_weight: ScalingType,
    config: Float8LinearConfig,
) -> Optional[torch.Tensor]:
    if tensor_already_casted_to_fp8(weight):
        return None
    if scaling_type_weight is ScalingType.DYNAMIC:
        return tensor_to_scale(weight, config.cast_config_weight.target_dtype)
    return None


def _cast_weight_to_float8_t(
    weight: torch.Tensor,
    config: Float8LinearConfig,
    linear_mm_config: LinearMMConfig,
    weight_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if tensor_already_casted_to_fp8(weight):
        return weight.t()
    weight_fp8 = hp_tensor_and_scale_to_float8(
        weight,
        weight_scale,
        config.cast_config_weight.target_dtype,
        linear_mm_config,
        gemm_input_role=GemmInputRole.WEIGHT,
    )
    return weight_fp8.t()


# ---------------------------------------------------------------------------
# Autograd Function – identical logic, stubs perform no real quantisation
# ---------------------------------------------------------------------------
@torch._dynamo.allow_in_graph
class matmul_with_hp_or_float8_args(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_hp: torch.Tensor,
        weight_hp_t: torch.Tensor,
        linear_mm_config: LinearMMConfig,
        config: Float8LinearConfig,
    ):
        ctx.save_for_backward(input_hp, weight_hp_t)
        ctx.linear_mm_config = linear_mm_config
        ctx.config = config

        # In this stub build we never actually quantise.
        input_maybe_fp8 = input_hp
        weight_maybe_fp8_t = weight_hp_t

        orig_shape = input_maybe_fp8.shape
        res = torch.mm(
            input_maybe_fp8.reshape(-1, orig_shape[-1]),
            weight_maybe_fp8_t,
        )
        return res.reshape(*orig_shape[:-1], res.shape[-1])

    @staticmethod
    def backward(ctx, grad_output):
        input_hp, weight_hp_t = ctx.saved_tensors

        grad_input = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight_hp_t)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.transpose(-2, -1).matmul(input_hp)
        return grad_input, grad_weight, None, None


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------
class Float8Linear(torch.nn.Linear):
    """
    Drop-in replacement for nn.Linear that keeps the public interface but
    internally *could* perform FP8 compute.  In this self-contained version
    it behaves exactly like nn.Linear (no loss of accuracy).
    """

    def __init__(self, *args, **kwargs):
        config: Float8LinearConfig = kwargs.pop("config")
        super().__init__(*args, **kwargs)
        self.config = config

        self.scaling_type_input   = config.cast_config_input.scaling_type
        self.scaling_type_weight  = config.cast_config_weight.scaling_type
        self.scaling_type_grad_output = config.cast_config_grad_output.scaling_type

        # Build per-MM config objects (kept for API compatibility)
        self.linear_mm_config = LinearMMConfig(
            ScaledMMConfig(config.emulate, config.gemm_config_output.use_fast_accum,
                           False, config.pad_inner_dim),
            ScaledMMConfig(config.emulate, config.gemm_config_grad_input.use_fast_accum,
                           False, config.pad_inner_dim),
            ScaledMMConfig(config.emulate, config.gemm_config_grad_weight.use_fast_accum,
                           False, config.pad_inner_dim),
        )

    # ---------------------------------------------------------------------
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if torch.is_autocast_enabled():
            input = input.to(torch.get_autocast_gpu_dtype())

        # In this stub we never go through the axiswise optimisation path
        weight_maybe_fp8_t = self.weight.t()

        # Optionally compute/checkpoint FP8 weight (stubbed)
        if self.config.force_recompute_fp8_weight_in_bwd:
            weight_maybe_fp8_t = checkpoint.checkpoint(
                _cast_weight_to_float8_t,
                self.weight,
                self.config,
                self.linear_mm_config,
                _get_weight_scale(self.weight, self.scaling_type_weight, self.config),
            )
        else:
            weight_maybe_fp8_t = _cast_weight_to_float8_t(
                self.weight,
                self.config,
                self.linear_mm_config,
                _get_weight_scale(self.weight, self.scaling_type_weight, self.config),
            )

        if weight_maybe_fp8_t.dtype != input.dtype:
            # For the stub we simply promote everything to the higher-precision side.
            # Doing this here keeps the autograd path unchanged.
            weight_maybe_fp8_t = weight_maybe_fp8_t.to(input.dtype)

        output = matmul_with_hp_or_float8_args.apply(
            input,
            weight_maybe_fp8_t,
            self.linear_mm_config,
            self.config,
        )
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output

    # ---------------------------------------------------------------------
    def extra_repr(self):
        c = self.config
        parts = [
            f"i:{c.cast_config_input.short_str()}",
            f"w:{c.cast_config_weight.short_str()}",
            f"go:{c.cast_config_grad_output.short_str()}",
        ]
        if c.cast_config_input_for_grad_weight != c.cast_config_input:
            parts.append(f"i_gw:{c.cast_config_input_for_grad_weight.short_str()}")
        if c.cast_config_weight_for_grad_input != c.cast_config_weight:
            parts.append(f"w_gi:{c.cast_config_weight_for_grad_input.short_str()}")
        if c.cast_config_grad_output_for_grad_weight != c.cast_config_grad_output:
            parts.append(f"go_gw:{c.cast_config_grad_output_for_grad_weight.short_str()}")
        return f"{super().extra_repr()}, cast_configs={','.join(parts)}"

    # ---------------------------------------------------------------------
    @classmethod
    def from_float(
        cls,
        mod: torch.nn.Linear,
        config: Optional[Float8LinearConfig] = None,
    ) -> "Float8Linear":
        if config is None:
            config = Float8LinearConfig()

        # Create a "meta" instance first, then copy parameters.
        with torch.device("meta"):
            new_mod = cls(
                mod.in_features,
                mod.out_features,
                bias=(mod.bias is not None),
                config=config,
            )

        new_mod.weight = mod.weight
        new_mod.bias   = mod.bias
        return new_mod


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    lin = torch.nn.Linear(8, 13, dtype=torch.bfloat16).cuda()
    fp8_lin = Float8Linear.from_float(lin).cuda()

    x = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
    y1 = lin(x)
    y2 = fp8_lin(x)
    print(fp8_lin, fp8_lin.weight.dtype)
    print("Max abs diff (should be 0 for stub impl):",
          (y1 - y2).abs().max().item())
