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

from abc import ABC, abstractmethod
from typing import Any, Optional

try:
    from torch.distributed.device_mesh import DeviceMesh
except ImportError:
    DeviceMesh = None


class StateDictAdapter(ABC):
    """Abstract base class for state dict transformations.

    This class defines the interface for converting between native model
    state dict format and other model state dict formats.
    """

    @abstractmethod
    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert from native model state dict to HuggingFace format.

        Args:
            state_dict: The native model state dict

        Returns:
            The converted HuggingFace format state dict
        """
        pass

    @abstractmethod
    def from_hf(
        self, hf_state_dict: dict[str, Any], device_mesh: Optional["DeviceMesh"] = None, target_format: str = "auto"
    ) -> dict[str, Any]:
        """Obtain native model state dict from HuggingFace format.

        Args:
            hf_state_dict: The HuggingFace format state dict
            device_mesh: Optional device mesh for DTensor expert parallelism.
                        If provided, only loads experts needed for the current rank.
            target_format: Target format for the conversion ("auto", "grouped_experts" or "deepep")

        Returns:
            The converted native model state dict
        """
        pass

    def get_expert_range_for_rank(self, device_mesh: Optional["DeviceMesh"], n_experts: int) -> tuple[int, int]:
        """Get the range of experts that should be loaded for the current rank.

        Args:
            device_mesh: Device mesh for expert parallelism
            n_experts: Total number of experts

        Returns:
            Tuple of (start_expert_id, end_expert_id) for this rank
        """
        if device_mesh is None or DeviceMesh is None:
            # No expert parallelism, load all experts
            return 0, n_experts

        # Get expert parallel dimension info
        ep_mesh = device_mesh["ep"] if "ep" in device_mesh.mesh_dim_names else device_mesh
        world_size = ep_mesh.size()
        rank = ep_mesh.get_rank()

        # Calculate expert range for this rank
        experts_per_rank = n_experts // world_size
        remainder = n_experts % world_size

        # Distribute remainder experts to first few ranks
        if rank < remainder:
            experts_per_rank += 1
            start_expert = rank * experts_per_rank
        else:
            start_expert = rank * experts_per_rank + remainder

        end_expert = start_expert + experts_per_rank
        return start_expert, end_expert

    def should_load_expert(self, expert_id: int, device_mesh: Optional["DeviceMesh"], n_experts: int) -> bool:
        """Check if a specific expert should be loaded on the current rank.

        Args:
            expert_id: The expert ID to check
            device_mesh: Device mesh for expert parallelism
            n_experts: Total number of experts

        Returns:
            True if this expert should be loaded on the current rank
        """
        start_expert, end_expert = self.get_expert_range_for_rank(device_mesh, n_experts)
        return start_expert <= expert_id < end_expert
