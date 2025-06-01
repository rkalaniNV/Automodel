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

from __future__ import annotations

import atexit
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
)

# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _safe_flatten(mesh: DeviceMesh, dim_names: Sequence[str], flat_name: str) -> None:
    """Flatten `mesh[dim_names]` into one new dimension named `flat_name`.

    Uses public `flatten` if available; falls back to private `_flatten`.
    """
    sub = mesh[tuple(dim_names)]
    if hasattr(sub, "flatten"):
        sub.flatten(mesh_dim_name=flat_name)
    else:  # PyTorch <2.2 uses a private method
        sub._flatten(mesh_dim_name=flat_name)  # type: ignore[attr-defined]

def _get_dim_pg(mesh: DeviceMesh, dim_name: str) -> dist.ProcessGroup:
    """
    Return the process-group corresponding to ``dim_name`` in ``mesh``.
    Works with both regular and flattened meshes and across PyTorch
    versions (≥2.0).
    """
    # 1. Direct attribute (PyTorch 2.2+ flattened mesh exposes it)
    sub = mesh[dim_name]
    if hasattr(sub, "get_group"):
        return sub.get_group()          # type: ignore[attr-defined]
    if hasattr(sub, "_process_group"):
        return sub._process_group               # type: ignore[attr-defined]

    # 2. Fall back to dim-group interface
    if hasattr(mesh, "get_dim_group"):          # 2.2+
        return mesh.get_dim_group(dim_name)
    if hasattr(mesh, "get_dim_groups"):         # <= 2.1
        idx = mesh.mesh_dim_names.index(dim_name)  # type: ignore[attr-defined]
        return mesh.get_dim_groups()[idx]       # type: ignore[attr-defined]

    raise RuntimeError(f"Cannot obtain process group for dim {dim_name}")

def get_world_size():
    if dist.is_available() and dist.is_initialized():
        ws = dist.get_world_size()
    else:
        # Try to read from env (torchrun sets it), else assume 1
        ws = int(os.environ.get("WORLD_SIZE", "1"))
    return ws

@dataclass(slots=True)
class ParallelDims:
    """Logical parallelism description.

    +----------------------------------------------------------+
    | dp_replicate | dp_shard | cp | tp | pp | ep | world_size |
    +----------------------------------------------------------+
    | 1            | 1        | 1  | 1  | 1  | 1  | 1          | single-device
    +----------------------------------------------------------+
    | >1           | 1        | 1  | 1  | 1  | 1  | >1         | DDP
    +----------------------------------------------------------+
    | 1            | >1       | 1  | 1  | 1  | 1  | >1         | FSDP
    +----------------------------------------------------------+
    | >1           | >1       | 1  | 1  | 1  | 1  | >1         | HSDP
    +----------------------------------------------------------+

    Each attribute is the degree of parallelism for that axis.
      dp_replicate : data-parallel replication (DDP semantics)
      dp_shard     : data-parallel sharding (ZeRO style)
      cp           : chunk / spatial tensor parallel
      tp           : tensor parallel (column/row)
      pp           : pipeline parallel
      ep           : expert parallel
      world_size   : total processes
      enable_loss_parallel : optional loss-parallel flag
    """

    dp_replicate: int = 1
    dp_shard: int = -1           # ‑1 means “infer from world_size”
    cp: int = 1
    tp: int = 1
    pp: int = 1
    ep: int = 1
    world_size: int = 1
    enable_loss_parallel: bool = False

    # internal cache for build_mesh result (filled by ParallelContext)
    _mesh: Optional[DeviceMesh] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate_and_infer()

    def _validate_and_infer(self) -> None:
        """Validate config and, when dp_shard == -1, infer its value."""
        # Check type
        for attr_name in ("dp_replicate", "dp_shard", "cp", "tp", "pp", "ep"):
            val = getattr(self, attr_name)
            if not isinstance(val, int):
                raise TypeError(f"{attr_name} must be int (got {type(val).__name__})")

        if self.world_size < 1:
            raise ValueError("world_size must be >= 1")

        # -------- single-GPU guard --------
        if self.world_size == 1:

            multi_dims = [(k, getattr(self, k))
                for k in
                ("dp_replicate", "dp_shard", "cp", "tp", "pp", "ep")
                if getattr(self, k) > 1
            ]
            if multi_dims:
                raise ValueError(
                    f"world_size==1 but parallel dims {multi_dims} > 1. "
                    "Either launch more ranks or set them to 1."
                )

        # -------- non-negative checks --------
        for n in ("dp_replicate", "cp", "tp", "pp", "ep"):
            val = getattr(self, n)
            if val < 1:
                raise ValueError(f"{n} must be >= 1 (got {val})")

        if self.dp_shard not in (-1, 1) and self.dp_shard < 1:
            raise ValueError("dp_shard must be -1 or >= 1")

        # -------- infer dp_shard if requested --------
        if self.dp_shard == -1:
            denom = self.dp_replicate * self.cp * self.tp * self.pp
            if self.world_size % denom:
                raise ValueError(
                    f"world_size={self.world_size} not divisible by "
                    f"dp_replicate*cp*tp*pp={denom} (cannot infer dp_shard)"
                )
            self.dp_shard = self.world_size // denom

        # -------- per-dim divisibility / range --------
        for n in ("dp_replicate", "dp_shard", "cp", "tp", "pp", "ep"):
            v = getattr(self, n)
            if v > self.world_size:
                raise ValueError(f"{n} ({v}) cannot exceed world_size ({self.world_size})")
            if self.world_size % v:
                raise ValueError(f"world_size ({self.world_size}) must be divisible by {n} ({v})")

        # -------- final product check --------
        total = (
            self.dp_replicate * self.dp_shard *
            self.cp * self.tp * self.pp * self.ep
        )
        if total != self.world_size:
            raise ValueError(
                f"Product of dims ({total}) != world_size ({self.world_size})"
            )

    # ---------------- Convenience flags ---------------- #
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
    def cp_enabled(self) -> bool:
        return self.cp > 1

    @property
    def tp_enabled(self) -> bool:
        return self.tp > 1

    @property
    def pp_enabled(self) -> bool:
        return self.pp > 1

    @property
    def ep_enabled(self) -> bool:
        return self.ep > 1

    @property
    def loss_parallel_enabled(self) -> bool:
        return self.tp > 1 and self.enable_loss_parallel

    @property
    def non_data_parallel_size(self) -> int:
        return self.cp * self.tp * self.pp * self.ep

    # TODO(akoumparouli): switch to enum instead of is_ddp/is_single_device
    @property
    def is_ddp(self):
        return self.non_data_parallel_size == 1 and self.dp_shard == 1 and self.dp_replicate > 1

    @property
    def is_single_device(self):
        return self.non_data_parallel_size == 1 and self.dp_shard == 1 and self.dp_replicate == 1

    # ---------------- Mesh builder ---------------- #
    def build_mesh(self, device_type: str = "cuda") -> DeviceMesh:
        """Construct a DeviceMesh containing only dims with size>1."""
        dims, names = [], []
        for size, n in (
            (self.dp_replicate, "dp_replicate"),
            (self.dp_shard, "dp_shard"),
            (self.pp, "pp"),
            (self.cp, "cp"),
            (self.tp, "tp"),
            (self.ep, "ep"),
        ):
            if size > 1:
                dims.append(size)
                names.append(n)

        if not dims:  # all ones → 0-D mesh
            logger.info("All parallel dims ==1 → degenerate 0-D mesh")
            return init_device_mesh(device_type, [])  # empty mesh

        logger.info("Building %d-D device mesh: names=%s, sizes=%s", len(dims), names, dims)
        mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)

        # Flatten commonly-used composite meshes so that downstream code
        # can obtain a dedicated process group quickly.
        dp_names, dp_shard_cp_names, dp_cp_names = [], [], []

        if self.dp_replicate_enabled:
            dp_names.append("dp_replicate")
            dp_cp_names.append("dp_replicate")
        if self.dp_shard_enabled:
            dp_names.append("dp_shard")
            dp_shard_cp_names.append("dp_shard")
            dp_cp_names.append("dp_shard")
        if self.cp_enabled:
            dp_shard_cp_names.append("cp")
            dp_cp_names.append("cp")

        if dp_names:
            _safe_flatten(mesh, dp_names, "dp")
        if dp_shard_cp_names:
            _safe_flatten(mesh, dp_shard_cp_names, "dp_shard_cp")
        if dp_cp_names:
            _safe_flatten(mesh, dp_cp_names, "dp_cp")

        return mesh


class ParallelContext:
    """Runtime object returned by `init_parallel`.

    Attributes
    ----------
    rank : int
    world_size : int
    dims : ParallelDims
    mesh : Optional[DeviceMesh]     (None in single-GPU mode)
    ddp_pg : Optional[dist.ProcessGroup]  (None in single-GPU mode)
    """

    def __init__(
        self,
        dims: ParallelDims,
        backend: str = "nccl",
        device_type: str = "cuda",
        init_pg_kwargs: Optional[dict] = {},
    ):
        self.dims = dims
        self._destroyed: bool = False     # internal flag
        atexit.register(self.shutdown)

        # 1. Distributed init (if not yet initialized)
        if torch.cuda.is_available() and device_type == "cuda":
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

        if dist.is_available() and not dist.is_initialized() and dims.world_size > 1:
            logger.info("Initializing default process group (%s)...", backend)
            dist.init_process_group(backend=backend, **init_pg_kwargs)

        # 2. Build mesh
        self.mesh = dims.build_mesh(device_type)

    def shutdown(self) -> None:
        """Idempotent cleanup of the default PG (called at exit)."""
        if not self._destroyed:
            if dist.is_available() and dist.is_initialized():
                logger.info("Destroying default process group")
                dist.destroy_process_group()
            self._destroyed = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False   # don’t swallow exceptions

# --------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------- #
def init_parallel(
    dp_replicate: int = 1,
    dp_shard: int = -1,
    cp: int = 1,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    world_size: int = None,
    enable_loss_parallel: bool = False,
    backend: str = "nccl",
    device_type: str = "cuda",
) -> ParallelContext:
    """Create a ParallelContext for single-GPU, DDP or model-parallel runs.

    Parameters mirror `ParallelDims`.  `world_size` is inferred from
    the initialized process group (or from environment variable
    WORLD_SIZE).  Call this once at program start and pass the resulting
    context to the rest of your code.
    """
    if world_size is None:
        world_size = get_world_size()
    if dp_replicate < 1:
        dp_replicate = world_size
    dims = ParallelDims(
        dp_replicate=dp_replicate,
        dp_shard=dp_shard,
        cp=cp,
        tp=tp,
        pp=pp,
        ep=ep,
        world_size=world_size,
        enable_loss_parallel=enable_loss_parallel,
    )
    return ParallelContext(dims, backend=backend, device_type=device_type)


# --------------------------------------------------------------------- #
# Simple CLI test
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    # Example: python -m torch.distributed.run --nproc_per_node=4 parallel.py
    ctx = init_parallel(dp_replicate=1, dp_shard=1)  # pure DDP across 4 GPUs
    logger.info(
        "Rank %d/%d | mesh=%s | dp_pg_size=%s",
        ctx.rank,
        ctx.world_size,
        ctx.mesh,
        ctx.ddp_pg.size() if ctx.ddp_pg is not None else 1,
    )
