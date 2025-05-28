import os
import re
import shutil
import atexit
from pathlib import Path
from contextlib import contextmanager

from typing import Any, Dict, Union, Optional
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor._api import distribute_tensor
from torch.distributed.tensor.placement_types import Replicate, Shard
from automodel.utils.import_utils import safe_import_from
from dataclasses import dataclass, field

MixedPrecisionPolicy, HAS_MIXED_PRECISION_POLICY = safe_import_from(
    "torch.distributed.fsdp", "MixedPrecisionPolicy", fallback_module="torch.distributed._composable.fsdp"
)
fully_shard, HAS_FULLY_SHARD = safe_import_from(
    "torch.distributed.fsdp", "fully_shard", fallback_module="torch.distributed._composable.fsdp"
)
CPUOffloadPolicy, HAS_CPU_OFFLOAD_POLICY = safe_import_from(
    "torch.distributed.fsdp", "CPUOffloadPolicy", fallback_module="torch.distributed._composable.fsdp"
)
from automodel.distributed.parallelizer import fsdp2_strategy_parallelize



@dataclass
class CheckpointIO:
    """Simple disk-based checkpoint I/O."""
    def save(self, state: Dict[str, Any], path: Union[str, Path]):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, p)

    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        return torch.load(path, map_location="cpu")


@dataclass
class FSDP2Manager:
    dp_size: Optional[int] = field(
        default=None,
        metadata={"help": "Data‐parallel group size; if None, infer from WORLD_SIZE."}
    )
    tp_size: Optional[int] = field(
        default=None,
        metadata={"help": "Tensor‐parallel group size; if None, defaults to 1."}
    )
    cp_size: int = field(
        default=1,
        metadata={"help": "Context‐parallel group size (for pipeline‐like sharding)."}
    )
    sequence_parallel: bool = field(
        default=False,
        metadata={"help": "Enable sequence parallelism in TP plan if True."}
    )
    mp_policy: MixedPrecisionPolicy = field(
        default=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                output_dtype=torch.bfloat16,
                cast_forward_inputs=True,
            ),
        metadata={"help": "MixedPrecisionPolicy for FSDP2 (param/reduce/output dtypes)."}
    )
    offload_policy: CPUOffloadPolicy = field(
        default=None,
        metadata={"help": "CPUOffloadPolicy to offload parameters/optim states to CPU."}
    )
    backend: str = field(
        default="nccl",
        metadata={"help": "Distributed backend, e.g. 'nccl' or 'gloo'."}
    )

    _device_mesh: Any = field(
        default=None,
        init=False,
        metadata={"help": "Torch distributed DeviceMesh."}
    )
    _rank: int = field(
        default=None,
        init=False,
        metadata={"help": "Global rank of this process."}
    )
    world_size: int = field(
        default=None,
        init=False,
        metadata={"help": "Total number of processes."}
    )

    def __post_init__(self):
        # Ensure any string representations of dtypes coming from YAML are converted to actual torch.dtypes.
        def _to_torch_dtype(x):
            """Convert string like 'torch.float32' or 'float32' to torch.float32 constant."""
            if isinstance(x, str):
                # Strip optional leading module name
                if x.startswith("torch."):
                    x = x.split(".", 1)[1]
                try:
                    return getattr(torch, x)
                except AttributeError:
                    raise ValueError(f"Unsupported dtype string '{x}' in mp_policy configuration")
            return x

        if isinstance(self.mp_policy, MixedPrecisionPolicy):
            # Recreate policy only if any field is a string
            needs_fix = any(isinstance(getattr(self.mp_policy, f), str) for f in ("param_dtype", "reduce_dtype", "output_dtype"))
            if needs_fix:
                self.mp_policy = MixedPrecisionPolicy(
                    param_dtype=_to_torch_dtype(self.mp_policy.param_dtype),
                    reduce_dtype=_to_torch_dtype(self.mp_policy.reduce_dtype),
                    output_dtype=_to_torch_dtype(self.mp_policy.output_dtype),
                    cast_forward_inputs=self.mp_policy.cast_forward_inputs,
                )

        return self._setup_distributed()

    def _setup_distributed(self):
        """
        Initialize torch.distributed process group, infer dp/tp sizes,
        build device mesh, and register destroy handler.
        Requires env vars: RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT.
        """
        if not dist.is_available():
            raise RuntimeError("torch.distributed not available")

        if not dist.is_initialized():
            # raise RuntimeError("expected torch.distributed to be initialized")
            rank = int(os.environ["RANK"])
            world = int(os.environ["WORLD_SIZE"])
            os.environ.setdefault("MASTER_ADDR", os.environ.get("MASTER_ADDR", "localhost"))
            os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT", "29500"))
            dist.init_process_group(self.backend, rank=rank, world_size=world)

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        # infer if not provided
        self.dp_size = self.dp_size
        if self.dp_size is None or self.dp_size <= 0:
            self.dp_size = self.world_size
        self.tp_size = self.tp_size or 1

        # build mesh [dp, cp, tp]
        mesh_shape = (self.dp_size, self.cp_size, self.tp_size)
        mesh_names = ("data_parallel", "context_parallel", "tensor_parallel")
        self.device_mesh = init_device_mesh(
            device_type="cuda" if self.backend == "nccl" else "cpu",
            mesh_shape=mesh_shape,
            mesh_dim_names=mesh_names,
        )
        # flatten dp+cp if cp>1
        if self.cp_size > 1:
            self.device_mesh[("data_parallel", "context_parallel")]._flatten(mesh_dim_name="dp_cp")

        # move base model to the right device before wrapping
        # dev = torch.device("cuda" if self.backend=="nccl" else "cpu")
        # self.raw_model.to(dev)
        # self.model = self.raw_model  # will be replaced on parallelize()
        return self

    def parallelize(self, model):
        """
        Apply FSDP2 + TP sharding via the provided parallelize_fn.
        Must be called after setup_distributed().
        """
        fsdp2_strategy_parallelize(
            model,
            device_mesh=self.device_mesh,
            mp_policy=self.mp_policy,
            use_hf_tp_plan=False, #self.use_hf_tp,
            tp_shard_plan=None, #self.tp_plan,
            offload_policy=self.offload_policy,
        )
        return model

    @contextmanager
    def tensor_init_context(self):
        """
        Context manager for parameter init (e.g. empty_init).
        Usage:
            with manager.tensor_init_context():
                initialize_weights(...)
        """
        yield

    # def save_checkpoint(self, state: Dict[str, Any], path: Union[str, Path]):
    #     """
    #     Unshard + barrier + save on rank0.
    #     state: e.g. {"model": full_state_dict, "opt": optimizer.state_dict()}
    #     """
    #     dist.barrier()
    #     if self.rank == 0:
    #         self.checkpoint_io.save(state, path)

    # def load_checkpoint(self, path: Union[str, Path]) -> Dict[str, Any]:
    #     """
    #     Load checkpoint on rank0 + broadcast tensors to others.
    #     Returns the raw loaded state dict.
    #     """
    #     state = self.checkpoint_io.load(path)
    #     for k, v in state.items():
    #         if torch.is_tensor(v):
    #             dist.broadcast(v, src=0)
    #     return state

    # def shard_and_load_model(self, full_state: Dict[str, torch.Tensor], strict: bool = False):
    #     """
    #     Take a full CPU state_dict and redistribute (shard) it over the device_mesh,
    #     then call load_state_dict on the FSDP2‐wrapped model.
    #     """
    #     # collect plan keys by placement type
    #     col_keys = [k for k, p in (self.tp_plan or {}).items() if hasattr(p, "__class__") and p.__class__.__name__=="ColwiseParallel"]
    #     row_keys = [k for k, p in (self.tp_plan or {}).items() if hasattr(p, "__class__") and p.__class__.__name__=="RowwiseParallel"]
    #     seq_keys = [k for k, p in (self.tp_plan or {}).items() if hasattr(p, "__class__") and p.__class__.__name__=="SequenceParallel"]

    #     sharded: Dict[str, torch.Tensor] = {}
    #     for name, tensor in full_state.items():
    #         if any(re.match(pat, name) for pat in seq_keys):
    #             placements = (Shard(0), Replicate())
    #             mesh = self.device_mesh
    #         elif any(re.match(pat, name) for pat in col_keys):
    #             placements = (Shard(0), Shard(0))
    #             mesh = self.device_mesh
    #         elif any(re.match(pat, name) for pat in row_keys):
    #             placements = (Shard(0), Shard(1))
    #             mesh = self.device_mesh
    #         else:
    #             placements = (Shard(0),)
    #             mesh = self.device_mesh["data_parallel"]

    #         sharded[name] = distribute_tensor(tensor, mesh, placements=placements)

    #     self.model.load_state_dict(sharded, strict=strict)