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

from typing import cast

import torch
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpStrategy,
    PlacementStrategy,
    RuntimeSchemaInfo,
    StrategyType,
)
from torch.distributed.tensor._ops._tensor_ops import unshard_tensor_dim
from torch.distributed.tensor._ops.utils import (
    is_tensor_dim_sharded,
    normalize_dim,
    register_op_strategy,
)
from torch.distributed.tensor.placement_types import (
    Shard,
)

aten = torch.ops.aten


@register_op_strategy(aten.select.int, schema_info=RuntimeSchemaInfo(1))
def gen_select_strategy(op_schema: OpSchema) -> StrategyType:
    """
    In this select op, first determine the input specs, then determine the output specs.
    - Input specs:
        - If the input is sharded on the selected dim, unshard it and change to replicate.
        - Otherwise, keep the original input specs.
    - Output specs:
        - It checks the input specs with the following cases:
        - Case 1 shard_dim == selected_dim: not possible as the input is already unsharded.
        - Case 2 shard_dim < selected_dim: keep the input specs.
        - Case 3 shard_dim > selected_dim: shard_dim -= 1.
    """
    input_strategy = op_schema.args_schema[0]
    assert isinstance(input_strategy, OpStrategy)
    assert len(op_schema.args_schema) == 3
    selected_dim, index = (
        cast(int, op_schema.args_schema[1]),
        cast(int, op_schema.args_schema[2]),
    )
    input_shape = input_strategy.shape
    input_ndim = input_strategy.ndim
    selected_dim = normalize_dim(selected_dim, input_ndim)
    index = normalize_dim(index, input_shape[selected_dim])

    select_strategy = OpStrategy([])
    for arg_strategy in input_strategy.strategies:
        arg_spec = arg_strategy.output_spec

        # determine input spec
        input_specs = arg_spec
        if is_tensor_dim_sharded(arg_spec, dim=selected_dim):
            # if input is sharded on the selected dim, need to unshard it, change to replicate
            arg_target_placements = unshard_tensor_dim(arg_spec.placements, dim=selected_dim)
            input_specs = DTensorSpec(arg_spec.mesh, arg_target_placements)  # R

        # determine output spec
        output_specs = input_specs
        if input_specs.is_sharded():
            # handle cases with sharded_dim != selected_dim
            output_spec_placements = []
            for placement in input_specs.placements:
                if placement.is_shard():
                    shard_dim = cast(Shard, placement).dim
                    if shard_dim > selected_dim:
                        shard_dim -= 1
                    placement = Shard(dim=shard_dim)
                output_spec_placements.append(placement)
            output_specs = DTensorSpec(arg_spec.mesh, placements=tuple(output_spec_placements))

        select_strategy.strategies.append(
            PlacementStrategy(
                output_specs=output_specs,
                input_specs=(input_specs,),
            )
        )
    return select_strategy
