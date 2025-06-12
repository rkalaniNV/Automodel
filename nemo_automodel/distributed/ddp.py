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

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from dataclasses import dataclass, field
from nemo_automodel.distributed.distributed_inteface import DistributedInterface
import weakref

@dataclass
class DDPManager(DistributedInterface):
    """
    Manages setting up distributed training using PyTorch's distributed package and wraps a model
    with DistributedDataParallel (DDP).

    Attributes:
        backend (str): The distributed backend to use (e.g. "nccl" or "gloo"). Defaults to "nccl".
        rank (int): Global rank of this process. This is set during distributed setup.
        world_size (int): Total number of processes in the distributed group. Set at distributed setup.
    """
    backend: str = field(
        default="nccl",
        metadata={"help": "Distributed backend, e.g. 'nccl' or 'gloo'."}
    )

    world_size: int = field(
        default_factory=lambda: int,
        metadata={"help": "Total number of distributed processes."}
    )

    # This is populated in setup_distributed(), not by user:
    rank: int = field(
        init=False,
        default_factory=lambda: int,
        metadata={"help": "Global rank of this process."}
    )

    def __post_init__(self):
        """
        Post-initialization hook that sets up the distributed environment.
        """
        return self._setup_distributed()

    def _setup_distributed(self):
        """
        """
        if not dist.is_available():
            raise RuntimeError("torch.distributed not available")

        if not dist.is_initialized():
            raise RuntimeError("expected torch.distributed to be initialized")

    def parallelize(self, model):
        """
        Wraps the given model with DistributedDataParallel (DDP).

        Moves the model to the initialized device before wrapping. For CUDA devices,
        the device id is passed to DDP as device_ids; for CPU, no device ids are provided.

        Args:
            model (torch.nn.Module): The PyTorch model to be wrapped.

        Returns:
            torch.nn.parallel.DistributedDataParallel: The DDP-wrapped model.
        """
        if self.backend == 'nccl':
            device = torch.cuda.current_device()
            device_ids = [device]
        else:
            device = torch.cpu.current_device()
            device_ids = None
        ans = DDP(model.to(device), device_ids=device_ids)
        self.model = weakref.ref(ans)
        return ans


    @contextmanager
    def no_sync(self):
        """
        Context manager to temporarily disable gradient synchronization during backpropagation.

        This can be used for gradient accumulation:
            with manager.no_sync():
                loss.backward()

        When used within a DDP-wrapped model, it skips the gradient all‐reduce.
        """
        if isinstance(self.model, DDP):
            with self.model.no_sync():
                yield
        else:
            yield
