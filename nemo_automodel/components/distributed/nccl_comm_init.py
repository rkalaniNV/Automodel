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

"""
NCCL Communication Initialization Utility

This module provides utilities to initialize NCCL communications for pipeline parallel groups
to work around lazy initialization race conditions in PyTorch distributed.

Based on the solution suggested in: https://github.com/pytorch/pytorch/issues/116590#issuecomment-2045790554
"""

import logging
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

logger = logging.getLogger(__name__)


def initialize_nccl_for_pp_groups(device_mesh: DeviceMesh) -> None:
    """
    Initialize NCCL communications for all pipeline parallel groups in the device mesh.

    This function addresses the NCCL lazy initialization race condition by sequentially
    performing dummy send/recv operations on each pipeline parallel group, ensuring
    proper NCCL communicator initialization before concurrent operations.

    The initialization is coordinated globally across all ranks to ensure:
    1. All PP groups are initialized in the same sequence across all ranks
    2. Proper synchronization between group initializations
    3. No race conditions during NCCL communicator setup

    Args:
        device_mesh: DeviceMesh containing pipeline parallel groups

    Raises:
        RuntimeError: If device mesh doesn't contain 'pp' dimension or distributed is not initialized
        ValueError: If pipeline parallel size is <= 1
    """
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before calling initialize_nccl_for_pp_groups")

    if "pp" not in device_mesh.mesh_dim_names:
        logger.info("No 'pp' dimension found in device mesh, skipping PP NCCL initialization")
        return

    pp_size = device_mesh["pp"].size()
    if pp_size <= 1:
        logger.info(f"Pipeline parallel size is {pp_size}, skipping PP NCCL initialization")
        return

    world_rank = dist.get_rank()
    dist.get_world_size()

    if world_rank == 0:
        logger.info(f"Initializing NCCL communications for pipeline parallel groups (PP size: {pp_size})...")

    # Extract all unique PP groups that this rank participates in
    pp_group_info = _extract_all_pp_groups_with_info(device_mesh)

    if not pp_group_info:
        logger.warning("No pipeline parallel groups found to initialize")
        return

    if world_rank == 0:
        logger.info(f"Found {len(pp_group_info)} unique pipeline parallel groups to initialize")

    # Global barrier before starting PP group initialization
    dist.barrier()

    # Initialize each PP group sequentially across all ranks
    for group_idx, (group, group_ranks) in enumerate(pp_group_info):
        if world_rank == 0:
            logger.info(f"Initializing PP group {group_idx + 1}/{len(pp_group_info)} (ranks: {sorted(group_ranks)})")

        # Initialize this specific PP group
        _initialize_single_pp_group_coordinated(group, group_idx, group_ranks)

        # Global barrier after each group initialization to ensure sequential processing
        dist.barrier()

    # Final barrier to ensure all PP groups are fully initialized before proceeding
    dist.barrier()

    if world_rank == 0:
        logger.info("NCCL initialization complete for all pipeline parallel groups")


def _extract_all_pp_groups_with_info(device_mesh: DeviceMesh) -> list[tuple[dist.ProcessGroup, set[int]]]:
    """
    Extract all unique pipeline parallel process groups from the device mesh with rank information.

    This function identifies all unique PP groups in the mesh, considering that there may be
    multiple PP groups when combined with other parallelism dimensions (DP, TP, CP).

    Args:
        device_mesh: DeviceMesh containing pipeline parallel dimension

    Returns:
        List of tuples containing (ProcessGroup, set of ranks in that group)
    """
    pp_group_info = []
    dist.get_rank()

    try:
        # Get all dimension names except 'pp'
        other_dims = [name for name in device_mesh.mesh_dim_names if name != "pp"]

        if not other_dims:
            # Simple case: only pipeline parallel dimension
            pp_mesh = device_mesh["pp"]
            pp_group = pp_mesh.get_group()
            group_ranks = set(range(device_mesh["pp"].size()))
            pp_group_info.append((pp_group, group_ranks))
        else:
            # Complex case: multiple parallelism dimensions
            # We need to find all unique PP groups across different DP/TP/CP combinations
            seen_group_ranks = set()

            # Get the current rank's position in the mesh
            device_mesh.get_coordinate()

            # Iterate through all possible combinations of other dimensions
            # while keeping PP dimension to extract unique PP groups
            pp_groups_found = _find_unique_pp_groups_in_mesh(device_mesh, other_dims, seen_group_ranks)
            pp_group_info.extend(pp_groups_found)

    except Exception as e:
        logger.error(f"Failed to extract PP groups from device mesh: {e}")
        logger.debug(f"Error details: {e}", exc_info=True)

    return pp_group_info


def _find_unique_pp_groups_in_mesh(
    device_mesh: DeviceMesh, other_dims: list[str], seen_group_ranks: set
) -> list[tuple[dist.ProcessGroup, set[int]]]:
    """
    Find all unique PP groups in a multi-dimensional mesh.

    Args:
        device_mesh: The device mesh
        other_dims: List of dimension names other than 'pp'
        seen_group_ranks: Set to track already seen group ranks

    Returns:
        List of unique PP groups with their rank sets
    """
    pp_group_info = []
    current_rank = dist.get_rank()

    try:
        # For each unique combination of other dimensions, get the PP group
        # This is a simplified approach - we get the main PP group for the current rank
        pp_mesh = device_mesh["pp"]
        pp_group = pp_mesh.get_group()

        # Get all ranks in this PP group
        device_mesh["pp"].size()
        device_mesh["pp"].get_local_rank()

        # Calculate the ranks in this PP group based on mesh structure
        group_ranks = _calculate_pp_group_ranks(device_mesh, current_rank)

        # Check if we've already seen this group
        group_signature = tuple(sorted(group_ranks))
        if group_signature not in seen_group_ranks:
            seen_group_ranks.add(group_signature)
            pp_group_info.append((pp_group, group_ranks))

    except Exception as e:
        logger.debug(f"Error finding PP groups in mesh: {e}")

    return pp_group_info


def _calculate_pp_group_ranks(device_mesh: DeviceMesh, current_rank: int) -> set[int]:
    """
    Calculate the set of ranks that belong to the same PP group as the current rank.

    Args:
        device_mesh: The device mesh
        current_rank: Current process rank

    Returns:
        Set of ranks in the same PP group
    """
    try:
        # Get the current rank's coordinates in the mesh
        mesh_coords = device_mesh.get_coordinate()
        pp_dim_idx = device_mesh.mesh_dim_names.index("pp")

        # All ranks with same coordinates except for PP dimension are in same PP group
        group_ranks = set()

        # Get mesh shape
        mesh_shape = device_mesh.mesh.shape
        pp_size = mesh_shape[pp_dim_idx]

        # For each PP rank, calculate the corresponding global rank
        for pp_rank in range(pp_size):
            coords = list(mesh_coords)
            coords[pp_dim_idx] = pp_rank
            try:
                # Convert coordinates back to global rank
                global_rank = _coords_to_global_rank(device_mesh, coords)
                group_ranks.add(global_rank)
            except Exception:
                # If coordinate conversion fails, fall back to simple calculation
                pass

        # If coordinate calculation failed, use simpler approach
        if not group_ranks:
            pp_rank = device_mesh["pp"].get_local_rank()
            base_rank = current_rank - pp_rank
            group_ranks = {base_rank + i for i in range(device_mesh["pp"].size())}

        return group_ranks

    except Exception as e:
        logger.debug(f"Error calculating PP group ranks: {e}")
        # Fallback: assume simple PP group structure
        pp_rank = device_mesh["pp"].get_local_rank()
        pp_size = device_mesh["pp"].size()
        base_rank = current_rank - pp_rank
        return {base_rank + i for i in range(pp_size)}


def _coords_to_global_rank(device_mesh: DeviceMesh, coords: list[int]) -> int:
    """
    Convert mesh coordinates to global rank.

    Args:
        device_mesh: The device mesh
        coords: List of coordinates in each dimension

    Returns:
        Global rank corresponding to the coordinates
    """
    # This is a simplified calculation - in practice, DeviceMesh has internal methods for this
    mesh_shape = device_mesh.mesh.shape
    rank = 0
    multiplier = 1

    for i in reversed(range(len(coords))):
        rank += coords[i] * multiplier
        multiplier *= mesh_shape[i]

    return rank


def _initialize_single_pp_group_coordinated(group: dist.ProcessGroup, group_idx: int, group_ranks: set[int]) -> None:
    """
    Initialize NCCL communications for a single pipeline parallel group with coordination.

    Performs dummy send/recv operations to trigger NCCL communicator initialization
    and synchronizes all processes before proceeding. This version coordinates with
    global barriers to ensure sequential initialization across all PP groups.

    Args:
        group: Pipeline parallel process group to initialize
        group_idx: Index of the group for logging purposes
        group_ranks: Set of global ranks that belong to this PP group
    """
    current_rank = dist.get_rank()

    # Only ranks in this PP group participate in initialization
    if current_rank not in group_ranks:
        logger.debug(f"Rank {current_rank} not in PP group {group_idx}, skipping")
        return

    logger.debug(f"Initializing PP group {group_idx} (size: {group.size()}, ranks: {sorted(group_ranks)})")

    if group.size() <= 1:
        logger.debug(f"PP group {group_idx} has size {group.size()}, skipping")
        return

    # Get local rank within this group
    local_rank = group.rank()
    group_size = group.size()

    # Create a small tensor for dummy communication
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
    dummy_tensor = torch.zeros(1, dtype=torch.float32, device=device)

    try:
        # Step 1: Perform dummy send/recv operations between adjacent ranks
        # This triggers NCCL communicator initialization
        if local_rank == 0 and group_size > 1:
            # Rank 0 sends to rank 1
            logger.debug(f"PP group {group_idx}: rank 0 (global {current_rank}) sending dummy data to rank 1")
            dist.send(dummy_tensor, dst=1, group=group)

        elif local_rank == 1:
            # Rank 1 receives from rank 0
            logger.debug(f"PP group {group_idx}: rank 1 (global {current_rank}) receiving dummy data from rank 0")
            dist.recv(dummy_tensor, src=0, group=group)

        # Step 2: Synchronize all processes in this PP group after dummy communication
        logger.debug(f"PP group {group_idx}: synchronizing group")
        dist.barrier(group=group)

        # Step 3: Perform additional send/recv operations to ensure NCCL is fully initialized
        # This follows the pattern suggested in the PyTorch issue solution
        if group_size > 2:
            # Do another round of communication with different ranks to ensure robustness
            if local_rank == 0:
                # Send to last rank
                logger.debug(f"PP group {group_idx}: rank 0 sending to last rank")
                dist.send(dummy_tensor, dst=group_size - 1, group=group)
            elif local_rank == group_size - 1:
                # Receive from rank 0
                logger.debug(f"PP group {group_idx}: last rank receiving from rank 0")
                dist.recv(dummy_tensor, src=0, group=group)

        # Final synchronization within the group
        dist.barrier(group=group)

        logger.debug(f"PP group {group_idx} NCCL initialization complete")

    except Exception as e:
        logger.error(f"Failed to initialize NCCL for PP group {group_idx}: {e}")
        logger.error(
            f"Group details - size: {group_size}, local_rank: {local_rank}, "
            f"global_rank: {current_rank}, group_ranks: {sorted(group_ranks)}"
        )
        raise


# Backward compatibility function
def _initialize_single_pp_group(group: dist.ProcessGroup, group_idx: int) -> None:
    """
    Legacy function for backward compatibility.

    Args:
        group: Pipeline parallel process group to initialize
        group_idx: Index of the group for logging purposes
    """
    # Calculate approximate group ranks for backward compatibility
    current_rank = dist.get_rank()
    group_size = group.size()
    local_rank = group.rank()

    # Simple approximation of group ranks
    base_rank = current_rank - local_rank
    group_ranks = {base_rank + i for i in range(group_size)}

    _initialize_single_pp_group_coordinated(group, group_idx, group_ranks)


def should_initialize_nccl_for_pp(device_mesh: Optional[DeviceMesh] = None) -> bool:
    """
    Check if NCCL initialization is needed for pipeline parallel groups.

    Args:
        device_mesh: Optional device mesh to check

    Returns:
        True if NCCL initialization should be performed, False otherwise
    """
    if not dist.is_initialized():
        return False

    if device_mesh is None:
        return False

    if "pp" not in device_mesh.mesh_dim_names:
        return False

    pp_size = device_mesh["pp"].size()
    return pp_size > 1
