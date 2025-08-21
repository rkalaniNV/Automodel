# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import re
from typing import Any, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.moe.state_dict_utils import (
    create_dtensor_from_local,
    get_expert_range_for_rank_from_mesh,
    is_dtensor,
    should_load_expert_for_rank,
    split_experts_weights_dtensor_aware,
    validate_dtensor_expert_sharding,
)


class MoEStateDictMixin:
    """Mixin class providing MoE state dict conversion utilities.

    This mixin provides methods for:
    - Expert parallelism calculations (ranges, assignment)
    - Format conversion between HuggingFace and native formats
    - Both GroupedExperts and DeepEP format support
    - DTensor-aware expert loading and conversion

    Can be used by any MoE model that needs expert parallelism and format conversion.
    """

    # These attributes must be set by subclasses in their __init__ method:
    # - self.moe_config: MoE configuration object with expert settings
    # - self.config: Model configuration object
    # - self.backend: Backend configuration object

    # Expert parallelism utilities
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

    def _validate_expert_availability(
        self,
        hf_state_dict: dict[str, Any],
        n_experts: int,
        device_mesh: Optional["DeviceMesh"] = None,
    ) -> None:
        """Validate that all required experts are available in the HF state dict before loading.
        Only validates experts needed for the current rank and layers present in the state dict.

        Args:
            hf_state_dict: HuggingFace format state dict
            n_experts: Total number of experts
            device_mesh: Optional device mesh for expert parallelism

        Raises:
            RuntimeError: If required expert weights are missing from the checkpoint
        """
        # Determine which experts need to be loaded for this rank
        if device_mesh is not None:
            start_expert, end_expert = get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
            required_experts = list(range(start_expert, end_expert))
            rank = device_mesh["ep"].get_rank() if "ep" in device_mesh.mesh_dim_names else device_mesh.get_rank()
            rank_info = f" (rank {rank})"
        else:
            required_experts = list(range(n_experts))
            rank_info = ""

        # Detect key format and find layers that have expert weights in the state dict
        uses_model_prefix = any(key.startswith("model.") for key in hf_state_dict.keys() if ".mlp.experts." in key)
        key_prefix = "model." if uses_model_prefix else ""

        # Find which layers have expert weights in the state dict
        layers_with_experts = set()
        pattern = rf"{re.escape(key_prefix)}layers\.(\d+)\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight"
        for key in hf_state_dict.keys():
            match = re.match(pattern, key)
            if match:
                layer_num = int(match.group(1))
                layers_with_experts.add(layer_num)

        if not layers_with_experts:
            # No expert weights found in state dict - might not be an MoE model
            return

        # Check availability for each required expert in layers that have experts
        missing_weights = []
        projection_types = ["gate_proj", "up_proj", "down_proj"]

        for layer_num in layers_with_experts:
            for expert_id in required_experts:
                for proj_type in projection_types:
                    expected_key = f"{key_prefix}layers.{layer_num}.mlp.experts.{expert_id}.{proj_type}.weight"
                    if expected_key not in hf_state_dict:
                        missing_weights.append(expected_key)

        if missing_weights:
            missing_count = len(missing_weights)
            total_required = len(required_experts) * len(layers_with_experts) * len(projection_types)
            raise RuntimeError(
                f"Expert weights missing from checkpoint{rank_info}: {missing_count}/{total_required} required weights not found. "
                f"Cannot load experts - checkpoint may be incomplete or corrupted. "
                f"Layers with experts: {sorted(layers_with_experts)}, Required experts: {required_experts}. "
                f"First few missing keys: {missing_weights[:5]}"
                + (f" (and {missing_count - 5} more)" if missing_count > 5 else "")
            )

    # Abstract helper methods that must be implemented by subclasses
    # Concrete helper methods for expert weight manipulation
    def _split_experts_weights(self, weight: torch.Tensor, n_experts: int) -> list[torch.Tensor]:
        """Split grouped expert weights into individual expert weights.
        For grouped expert weights with shape [n_experts, ...], split into n_experts tensors each with shape [...].
        Supports both regular tensors and DTensors.
        """
        if is_dtensor(weight):
            # Use DTensor-aware splitting
            split_weights, expert_ids = split_experts_weights_dtensor_aware(weight, n_experts)
            # For backward compatibility, store expert IDs for later use
            self._last_expert_ids = expert_ids
            return split_weights
        else:
            # Regular tensor handling
            if weight.shape[0] != n_experts:
                raise ValueError(f"Expected first dimension to be {n_experts}, got {weight.shape[0]}")

            # Split along first dimension (one tensor per expert) and squeeze out the expert dimension
            split_weights = []
            expert_ids = []
            for i in range(n_experts):
                expert_weight = weight[i]  # Shape: [...] (expert dimension removed)
                split_weights.append(expert_weight)
                expert_ids.append(i)

            self._last_expert_ids = expert_ids
            return split_weights

    def _concatenate_expert_weights(
        self, expert_weights_by_layer: dict[str, Any], n_experts: int
    ) -> Optional[torch.Tensor]:
        """Concatenate the weights of separate experts into GroupedExpert weights.

        Args:
            expert_weights_by_layer: Nested dict structure containing expert weights
            n_experts: Total number of experts expected

        Returns:
            Stacked tensor if all experts are available for a layer, None otherwise
        """
        for layer, abstract_keys in list(expert_weights_by_layer.items()):
            for abstract_key, experts in list(abstract_keys.items()):
                # If we have all the experts for this abstract_key, concatenate them
                if len(experts) == n_experts:
                    sorted_expert_ids = sorted(experts.keys())
                    sorted_experts = [experts[i] for i in sorted_expert_ids]
                    stacked_tensor = torch.stack(sorted_experts, dim=0)

                    # Remove these experts from the tracking dict to free memory
                    del expert_weights_by_layer[layer][abstract_key]
                    if not expert_weights_by_layer[layer]:
                        del expert_weights_by_layer[layer]

                    return stacked_tensor

        return None

    # Format conversion methods - concrete implementations for MoE models
    def _to_hf_grouped_experts(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert GroupedExperts format to HuggingFace format.
        Handles: gate_projs, up_projs, down_projs -> individual expert weights
        """

        n_experts = self.moe_config.n_routed_experts
        hf_state_dict: dict[str, Any] = {}

        # Handle GroupedExperts tensors
        for key, value in list(state_dict.items()):
            if ".mlp.experts.gate_projs" in key and key.endswith(".gate_projs"):
                # GroupedExperts: [n_experts, inter_dim, dim] -> per-expert gate_proj [inter_dim, dim]
                layer_num = re.search(r"layers\.(\d+)", key).group(1)

                # Validate DTensor expert sharding if applicable
                if is_dtensor(value):
                    validate_dtensor_expert_sharding(value, n_experts, f"gate_projs layer {layer_num}")

                splits = self._split_experts_weights(value, n_experts)
                # Use actual expert IDs from DTensor-aware splitting
                for i, w in enumerate(splits):
                    expert_id = self._last_expert_ids[i]
                    prefix = "model." if self._uses_model_prefix else ""
                    new_key = f"{prefix}layers.{layer_num}.mlp.experts.{expert_id}.gate_proj.weight"
                    hf_state_dict[new_key] = w
                continue

            if ".mlp.experts.up_projs" in key and key.endswith(".up_projs"):
                # GroupedExperts: [n_experts, inter_dim, dim]
                layer_num = re.search(r"layers\.(\d+)", key).group(1)

                # Validate DTensor expert sharding if applicable
                if is_dtensor(value):
                    validate_dtensor_expert_sharding(value, n_experts, f"up_projs layer {layer_num}")

                splits = self._split_experts_weights(value, n_experts)
                # Use actual expert IDs from DTensor-aware splitting
                for i, w in enumerate(splits):
                    expert_id = self._last_expert_ids[i]
                    prefix = "model." if self._uses_model_prefix else ""
                    new_key = f"{prefix}layers.{layer_num}.mlp.experts.{expert_id}.up_proj.weight"
                    hf_state_dict[new_key] = w
                continue

            if ".mlp.experts.down_projs" in key and key.endswith(".down_projs") and value.ndim == 3:
                # GroupedExperts: [n_experts, dim, inter_dim] -> per-expert down_proj [dim, inter_dim]
                layer_num = re.search(r"layers\.(\d+)", key).group(1)

                # Validate DTensor expert sharding if applicable
                if is_dtensor(value):
                    validate_dtensor_expert_sharding(value, n_experts, f"down_projs layer {layer_num}")

                splits = self._split_experts_weights(value, n_experts)
                # Use actual expert IDs from DTensor-aware splitting
                for i, w in enumerate(splits):
                    expert_id = self._last_expert_ids[i]
                    prefix = "model." if self._uses_model_prefix else ""
                    new_key = f"{prefix}layers.{layer_num}.mlp.experts.{expert_id}.down_proj.weight"
                    hf_state_dict[new_key] = w
                continue

        # Pass through non-expert keys unchanged
        for key, value in state_dict.items():
            if ".mlp.experts." in key and (
                key.endswith(".gate_projs") or key.endswith(".up_projs") or key.endswith(".down_projs")
            ):
                continue
            hf_state_dict[key] = value

        # Pass through - subclasses can override to add quantization scale tensors
        return hf_state_dict

    def _to_hf_deepep(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert DeepEP format to HuggingFace format.
        Handles: gate_and_up_projs, down_projs -> individual expert weights
        """

        n_experts = self.moe_config.n_routed_experts
        inter_dim = self.moe_config.moe_inter_dim
        hf_state_dict: dict[str, Any] = {}

        # Handle DeepEP tensors
        for key, value in list(state_dict.items()):
            if ".mlp.experts.gate_and_up_projs" in key and key.endswith(".gate_and_up_projs"):
                # DeepEP: [n_experts, dim, 2*inter_dim] -> per-expert gate_proj/up_proj [inter_dim, dim]
                layer_num = re.search(r"layers\.(\d+)", key).group(1)

                # Validate DTensor expert sharding if applicable
                if is_dtensor(value):
                    validate_dtensor_expert_sharding(value, n_experts, f"gate_and_up_projs layer {layer_num}")

                splits = self._split_experts_weights(value, n_experts)
                # Use actual expert IDs from DTensor-aware splitting
                for i, w in enumerate(splits):
                    expert_id = self._last_expert_ids[i]
                    # Split concat along last dim and transpose to [inter_dim, dim]
                    w_gate = w[:, :inter_dim].transpose(0, 1).contiguous()
                    w_up = w[:, inter_dim:].transpose(0, 1).contiguous()
                    prefix = "model." if self._uses_model_prefix else ""
                    hf_state_dict[f"{prefix}layers.{layer_num}.mlp.experts.{expert_id}.gate_proj.weight"] = w_gate
                    hf_state_dict[f"{prefix}layers.{layer_num}.mlp.experts.{expert_id}.up_proj.weight"] = w_up
                continue

            if (
                ".mlp.experts.down_projs" in key
                and key.endswith(".down_projs")
                and value.ndim == 3
                and value.shape[1] == inter_dim
            ):
                # DeepEP down: [n_experts, inter_dim, dim] -> per-expert down_proj [dim, inter_dim]
                layer_num = re.search(r"layers\.(\d+)", key).group(1)

                # Validate DTensor expert sharding if applicable
                if is_dtensor(value):
                    validate_dtensor_expert_sharding(value, n_experts, f"down_projs (DeepEP) layer {layer_num}")

                splits = self._split_experts_weights(value, n_experts)
                # Use actual expert IDs from DTensor-aware splitting
                for i, w in enumerate(splits):
                    expert_id = self._last_expert_ids[i]
                    prefix = "model." if self._uses_model_prefix else ""
                    hf_state_dict[f"{prefix}layers.{layer_num}.mlp.experts.{expert_id}.down_proj.weight"] = w.transpose(
                        0, 1
                    ).contiguous()
                continue

        # Pass through non-expert keys unchanged
        for key, value in state_dict.items():
            if ".mlp.experts." in key and (key.endswith(".gate_and_up_projs") or key.endswith(".down_projs")):
                continue
            hf_state_dict[key] = value

        # Pass through - subclasses can override to add quantization scale tensors
        return hf_state_dict

    def _from_hf_grouped_experts(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
    ) -> dict[str, Any]:
        """Convert HF checkpoint to GroupedExperts format.
        Creates separate gate_projs, up_projs, and down_projs tensors.
        """

        n_experts = self.moe_config.n_routed_experts

        # Validate that all required experts are available before loading
        self._validate_expert_availability(hf_state_dict, n_experts, device_mesh)

        # Determine which experts to load for this rank
        if device_mesh is not None:
            start_expert, end_expert = get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
            expected_experts_per_rank = end_expert - start_expert
            rank = device_mesh["ep"].get_rank() if "ep" in device_mesh.mesh_dim_names else device_mesh.get_rank()
            print(
                f"[DTensor Loading] Rank {rank}: loading experts {start_expert}-{end_expert - 1} ({expected_experts_per_rank}/{n_experts})"
            )
        else:
            start_expert, end_expert = 0, n_experts
            expected_experts_per_rank = n_experts
            rank = None

        state_dict: dict[str, Any] = {}
        expert_weights_by_layer: dict[str, dict[str, dict[int, torch.Tensor]]] = {}

        for key, value in hf_state_dict.items():
            if ".mlp.experts." in key and key.endswith(".weight"):
                # Handle both formats:
                # - model.layers.{L}.mlp.experts.{E}.gate_proj.weight (with model prefix)
                # - layers.{L}.mlp.experts.{E}.gate_proj.weight (without model prefix)
                m = re.match(
                    r"(?:model\.)?layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", key
                )
                if m is None:
                    # Unknown expert subkey, pass through
                    state_dict[key] = value
                    continue

                layer_num, expert_num, which = m.groups()
                expert_num = int(expert_num)

                # Skip experts not assigned to this rank
                if not should_load_expert_for_rank(expert_num, device_mesh, n_experts):
                    continue

                if layer_num not in expert_weights_by_layer:
                    expert_weights_by_layer[layer_num] = {}

                # Map HF key pattern to GroupedExperts format
                abstract_key = {
                    "gate_proj": "model.layers.{}.mlp.experts.gate_projs",
                    "up_proj": "model.layers.{}.mlp.experts.up_projs",
                    "down_proj": "model.layers.{}.mlp.experts.down_projs",
                }[which]

                native_key = abstract_key.format(layer_num)

                if native_key not in expert_weights_by_layer[layer_num]:
                    expert_weights_by_layer[layer_num][native_key] = {}

                # Store weights directly for GroupedExperts
                expert_weights_by_layer[layer_num][native_key][expert_num] = value

                # Check if we have all experts for this projection type
                if len(expert_weights_by_layer[layer_num][native_key]) == expected_experts_per_rank:
                    # Get expert IDs in sorted order (relative to this rank's range)
                    expert_ids = sorted(expert_weights_by_layer[layer_num][native_key].keys())
                    ordered = [expert_weights_by_layer[layer_num][native_key][i] for i in expert_ids]

                    # Shapes:
                    # gate/up: [inter_dim, dim] -> stacked [n_experts_this_rank, inter_dim, dim]
                    # down:    [dim, inter_dim] -> stacked [n_experts_this_rank, dim, inter_dim]
                    stacked = torch.stack(ordered, dim=0)

                    # Convert to DTensor if expert parallelism is enabled
                    dtensor = create_dtensor_from_local(stacked, device_mesh, rank)
                    state_dict[native_key] = dtensor

            else:
                # Pass through non-expert tensors
                if not key.endswith("_scale_inv"):
                    state_dict[key] = value

        return state_dict

    def _from_hf_deepep(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
    ) -> dict[str, Any]:
        """Convert HF checkpoint to DeepEP format.
        Creates combined gate_and_up_projs and transposed down_projs tensors.
        """

        n_experts = self.moe_config.n_routed_experts

        # Validate that all required experts are available before loading
        self._validate_expert_availability(hf_state_dict, n_experts, device_mesh)

        # Determine which experts to load for this rank
        if device_mesh is not None:
            start_expert, end_expert = get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
            expected_experts_per_rank = end_expert - start_expert
            rank = device_mesh["ep"].get_rank() if "ep" in device_mesh.mesh_dim_names else device_mesh.get_rank()
            print(
                f"[DTensor Loading] Rank {rank}: loading experts {start_expert}-{end_expert - 1} ({expected_experts_per_rank}/{n_experts})"
            )
        else:
            start_expert, end_expert = 0, n_experts
            expected_experts_per_rank = n_experts
            rank = None

        state_dict: dict[str, Any] = {}
        expert_weights_by_layer: dict[str, dict[str, dict[int, torch.Tensor]]] = {}

        for key, value in hf_state_dict.items():
            if ".mlp.experts." in key and key.endswith(".weight"):
                # Handle both formats:
                # - model.layers.{L}.mlp.experts.{E}.gate_proj.weight (with model prefix)
                # - layers.{L}.mlp.experts.{E}.gate_proj.weight (without model prefix)
                m = re.match(
                    r"(?:model\.)?layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", key
                )
                if m is None:
                    # Unknown expert subkey, pass through
                    state_dict[key] = value
                    continue

                layer_num, expert_num, which = m.groups()
                expert_num = int(expert_num)

                # Skip experts not assigned to this rank
                if not should_load_expert_for_rank(expert_num, device_mesh, n_experts):
                    continue

                if layer_num not in expert_weights_by_layer:
                    expert_weights_by_layer[layer_num] = {}

                # Map HF key pattern to DeepEP format
                if which in ["gate_proj", "up_proj"]:
                    native_key = f"model.layers.{layer_num}.mlp.experts.gate_and_up_projs"
                else:  # down_proj
                    native_key = f"model.layers.{layer_num}.mlp.experts.down_projs"

                if native_key not in expert_weights_by_layer[layer_num]:
                    expert_weights_by_layer[layer_num][native_key] = {}

                # Store weights with projection type info for DeepEP processing
                if which in ["gate_proj", "up_proj"]:
                    if expert_num not in expert_weights_by_layer[layer_num][native_key]:
                        expert_weights_by_layer[layer_num][native_key][expert_num] = {}
                    expert_weights_by_layer[layer_num][native_key][expert_num][which] = value

                    # Check if we have both gate and up for all experts
                    if len(expert_weights_by_layer[layer_num][native_key]) == expected_experts_per_rank and all(
                        isinstance(expert_data, dict) and "gate_proj" in expert_data and "up_proj" in expert_data
                        for expert_data in expert_weights_by_layer[layer_num][native_key].values()
                    ):
                        # Get expert IDs in sorted order
                        expert_ids = sorted(expert_weights_by_layer[layer_num][native_key].keys())

                        # Combine gate and up projections for DeepEP format
                        combined_tensors = []
                        for expert_id in expert_ids:
                            expert_data = expert_weights_by_layer[layer_num][native_key][expert_id]
                            gate_weight = expert_data["gate_proj"]  # [inter_dim, dim]
                            up_weight = expert_data["up_proj"]  # [inter_dim, dim]

                            # Transpose to [dim, inter_dim] and concatenate along last dim
                            gate_t = gate_weight.transpose(0, 1)  # [dim, inter_dim]
                            up_t = up_weight.transpose(0, 1)  # [dim, inter_dim]
                            combined = torch.cat([gate_t, up_t], dim=-1)  # [dim, 2*inter_dim]
                            combined_tensors.append(combined)

                        # Stack to [n_experts_this_rank, dim, 2*inter_dim]
                        stacked = torch.stack(combined_tensors, dim=0)

                        # Convert to DTensor if expert parallelism is enabled
                        dtensor = create_dtensor_from_local(stacked, device_mesh, rank)
                        state_dict[native_key] = dtensor

                else:  # down_proj
                    expert_weights_by_layer[layer_num][native_key][expert_num] = value

                    # Check if we have all down_proj experts
                    if len(expert_weights_by_layer[layer_num][native_key]) == expected_experts_per_rank:
                        # Get expert IDs in sorted order
                        expert_ids = sorted(expert_weights_by_layer[layer_num][native_key].keys())

                        # Transpose each expert's down_proj for DeepEP format
                        ordered = []
                        for expert_id in expert_ids:
                            down_weight = expert_weights_by_layer[layer_num][native_key][expert_id]  # [dim, inter_dim]
                            down_t = down_weight.transpose(0, 1)  # [inter_dim, dim]
                            ordered.append(down_t)

                        # Stack to [n_experts_this_rank, inter_dim, dim]
                        stacked = torch.stack(ordered, dim=0)

                        # Convert to DTensor if expert parallelism is enabled
                        dtensor = create_dtensor_from_local(stacked, device_mesh, rank)
                        state_dict[native_key] = dtensor

            else:
                # Pass through non-expert tensors
                if not key.endswith("_scale_inv"):
                    state_dict[key] = value

        return state_dict
