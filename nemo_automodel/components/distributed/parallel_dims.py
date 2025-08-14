# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
from dataclasses import dataclass
from functools import cached_property

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

log = logging.getLogger(__name__)


@dataclass
class ParallelDims:
    enable_loss_parallel: bool
    cp: int
    ep: int
    tp: int
    pp: int
    world_size: int
    dp_replicate: int
    dp_shard: int = -1
    dp_shard_with_ep: int = -1

    def __post_init__(self):
        self._validate()

    def _validate(self):
        cp, ep, tp, pp, world_size, dp_replicate, dp_shard, dp_shard_with_ep = (
            self.cp,
            self.ep,
            self.tp,
            self.pp,
            self.world_size,
            self.dp_replicate,
            self.dp_shard,
            self.dp_shard_with_ep,
        )
        for d in (cp, ep, tp, pp, dp_replicate, world_size):
            assert d >= 1, "Parallelism degree should be >= 1, except for dp_shard, dp_shard_with_ep."

        # Note:
        # 1) dp_replicate and dp_shard are used only for non-MoE weights.
        # 2) dp_shard_with_ep is only used for MoE weights.

        # Checks for MoE weights. Note that dp_shard is only used for default (non-MoE) weights.
        if ep > 1:
            assert dp_shard_with_ep == -1 or dp_shard_with_ep >= 1, " dp_shard_with_ep must -1 or >=1."
            if dp_shard_with_ep < 0:
                log.info(
                    "dp_shard_with_ep is set to -1, will be automatically determined based "
                    f"on world_size {world_size} // {pp * ep * tp}."
                )
                self.dp_shard_with_ep = dp_shard_with_ep = world_size // (pp * ep * tp)
                log.info(f"dp_shard_with_ep is set to {dp_shard_with_ep}.")
            assert (
                dp_shard_with_ep >= 1
            ), f"WORLD_SIZE({world_size}) is not a multiple of pp({pp}) * ep({ep}) * tp({tp})"

            if tp > 1:
                raise ValueError("tp must be 1 when ep > 1, since we only support MoE with pipeline parallelism.")

            if pp > 1 and dp_shard_with_ep > 1:
                raise ValueError(
                    "dp_shard_with_ep must be 1 when pp > 1, since we only support EP with pipeline parallelism."
                )

            if (pp * dp_shard_with_ep * ep * tp) != world_size:
                raise ValueError(
                    f"Invalid parallel dims: pp({pp}) * dp_shard_with_ep({dp_shard_with_ep}) * "
                    f"ep({ep}) * tp({tp}) != WORLD_SIZE({world_size})"
                )

        # Checks for default (non-MoE) weights.
        assert dp_shard == -1 or dp_shard >= 1, "dp_shard must -1 or >=1."
        if dp_shard < 0:
            log.info(
                "dp_shard is set to -1, will be automatically determined based on "
                f"world_size {world_size} // {pp * dp_replicate * cp * tp}."
            )
            self.dp_shard = dp_shard = world_size // (pp * dp_replicate * cp * tp)
            log.info(f"dp_shard is set to {dp_shard}.")
        assert dp_shard >= 1

        if pp > 1 and dp_replicate > 1:
            raise ValueError(
                "dp_replicate must be 1 when pp > 1, since we only support FSDP with pipeline parallelism."
            )

        if (pp * dp_replicate * dp_shard * cp * tp) != world_size:
            raise ValueError(
                f"Invalid parallel dims: pp({pp}) * dp_replicate({dp_replicate}) * "
                f"dp_shard({dp_shard}) * cp({cp}) * tp({tp}) != WORLD_SIZE({world_size})"
            )

    def _build_mesh(self, device_type: str, dims: list[int], names: list[str]) -> DeviceMesh:
        if len(dims) != len(names):
            raise ValueError("Dimensions and names must have the same length.")

        valid_dims = []
        valid_names = []
        for dim, name in zip(dims, names):
            if dim > 1:
                valid_dims.append(dim)
                valid_names.append(name)

        if math.prod(valid_dims) != self.world_size:
            raise ValueError(f"Invalid parallel dims: prod({valid_dims}) != WORLD_SIZE({self.world_size})")
        log.info(f"Building {len(valid_dims)}-D device mesh with {valid_names}, {valid_dims}")
        mesh = init_device_mesh(device_type, valid_dims, mesh_dim_names=valid_names)

        return mesh

    def build_meshes(self, device_type: str) -> dict[str, DeviceMesh]:
        meshes = {}

        meshes["default"] = self._build_mesh(
            device_type,
            [self.pp, self.dp_replicate, self.dp_shard, self.cp, self.tp],
            ["pp", "dp_replicate", "dp_shard", "cp", "tp"],
        )
        dp_shard_cp_mesh_dim_names = []
        if self.dp_shard_enabled:
            dp_shard_cp_mesh_dim_names.append("dp_shard")
        if self.cp_enabled:
            dp_shard_cp_mesh_dim_names.append("cp")
        if dp_shard_cp_mesh_dim_names != []:
            meshes["default"][tuple(dp_shard_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_shard_cp")

        if self.ep > 1:
            meshes["moe"] = self._build_mesh(
                device_type,
                [self.pp, self.dp_shard_with_ep, self.ep],
                ["pp", "dp_shard_with_ep", "ep"],
            )
        self._meshes = meshes
        return meshes

    @property
    def mesh(self) -> DeviceMesh:
        """Returns the default mesh."""
        if not hasattr(self, "_meshes"):
            raise ValueError("Meshes have not been built yet. Call `build_meshes` first.")
        return self._meshes["default"]

    @property
    def dp_enabled(self) -> bool:
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self) -> bool:
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self) -> bool:
        return self.dp_shard > 1

    @property
    def dp_shard_with_ep_enabled(self) -> bool:
        return self.dp_shard_with_ep > 1

    @property
    def cp_enabled(self) -> bool:
        return self.cp > 1

    @property
    def ep_enabled(self) -> bool:
        return self.ep > 1

    @property
    def tp_enabled(self) -> bool:
        return self.tp > 1

    @property
    def pp_enabled(self) -> bool:
        return self.pp > 1

    @property
    def loss_parallel_enabled(self) -> bool:
        return self.tp > 1 and self.enable_loss_parallel

    @cached_property
    def non_data_parallel_size(self) -> int:
        return self.cp * self.tp * self.pp
