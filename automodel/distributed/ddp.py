# ddp_utils.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from typing import Any, Dict, Union
from dataclasses import dataclass, field

@dataclass
class CheckpointIO:
    """Simple disk-based checkpoint I/O."""

    def save(self, state: Dict[str, Any], path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        return torch.load(path, map_location="cpu")


@dataclass
class DDPManager:
    checkpoint_io: CheckpointIO = field(
        default_factory=CheckpointIO,
        metadata={"help": "Helper for checkpoint save/load (disk, S3, etc.)."}
    )
    backend: str = field(
        default="nccl",
        metadata={"help": "Distributed backend, e.g. 'nccl' or 'gloo'."}
    )

    # These are populated in setup_distributed(), not by user:
    rank: int = field(
        init=False,
        default_factory=int,
        metadata={"help": "Global rank of this process."}
    )
    world_size: int = field(
        init=False,
        default_factory=int,
        metadata={"help": "Total number of distributed processes."}
    )

    def __post_init__(self):
        return self._setup_distributed()

    def _setup_distributed(self):
        """
        Initialize torch.distributed process group and wrap raw_model in DDP.
        Requires env vars: RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT.
        """
        if not dist.is_initialized():
            rank = int(os.environ["RANK"])
            world = int(os.environ["WORLD_SIZE"])
            os.environ.setdefault("MASTER_ADDR", os.environ.get("MASTER_ADDR", "localhost"))
            os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT", "29500"))
            dist.init_process_group(self.backend, rank=rank, world_size=world)

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # Pin GPU if using NCCL
        if self.backend == "nccl":
            local_gpu = self.rank % torch.cuda.device_count()
            torch.cuda.set_device(local_gpu)
            self.device = torch.device("cuda", index=local_gpu)
        else:
            self.device = torch.device("cpu")

    def parallelize(self, model):
        """Move the model to the correct device and wrap it with ``torch.nn.parallel.DistributedDataParallel``.

        The device is derived from the current global rank in the same way we do in
        ``setup_distributed``: for NCCL back-end we pin each rank to a single GPU
        (``rank % num_gpus``); otherwise we default to CPU.  This method used to
        reference an undefined variable ``device`` which caused a ``NameError`` at
        runtime – we recreate the exact device object here before using it.
        """
        model = model.to(self.device)
        # Wrap in DDP; for CPU or GLOO backend we pass no ``device_ids``.
        ddp_model = DDP(model, device_ids=[self.device] if self.device.type == "cuda" else None)
        return ddp_model

    # @contextmanager
    # def no_sync(self):
    #     """
    #     Context manager to skip gradient all‐reduce (for gradient accumulation):
    #         with manager.no_sync():
    #             ...
    #     """
    #     if isinstance(self.model, DDP):
    #         with self.model.no_sync():
    #             yield
    #     else:
    #         yield

    def save_checkpoint(self, state: Dict[str, Any], path: Union[str, Path]):
        """
        Barrier + save on rank 0.
        state: dict of tensors (e.g. {"model": model_state, "opt": opt_state})
        """
        dist.barrier()
        if self.rank == 0:
            self.checkpoint_io.save(state, path)

    def load_checkpoint(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load on rank 0 and broadcast any tensor entries to all ranks.
        Returns the state dict.
        """
        state = self.checkpoint_io.load(path)
        for k, v in state.items():
            if torch.is_tensor(v):
                dist.broadcast(v, src=0)
        return state