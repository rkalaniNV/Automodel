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
    current_rank = dist.get_rank()

    try:
        # Get the PP group for the current rank
        pp_mesh = device_mesh["pp"]
        pp_group = pp_mesh.get_group()

        # Get the actual ranks that are part of this PP group
        # This is the correct way to get the ranks in a process group
        group_ranks = _get_process_group_ranks(pp_group)

        logger.debug(f"Current rank {current_rank} is in PP group with ranks: {sorted(group_ranks)}")
        pp_group_info.append((pp_group, group_ranks))

    except Exception as e:
        logger.error(f"Failed to extract PP groups from device mesh: {e}")
        logger.debug(f"Error details: {e}", exc_info=True)

    return pp_group_info


def _get_process_group_ranks(group: dist.ProcessGroup) -> set[int]:
    """
    Get the set of global ranks that belong to a process group.

    Args:
        group: The process group

    Returns:
        Set of global ranks in the group
    """
    try:
        # Try to get ranks using the internal method if available
        if hasattr(group, "get_ranks"):
            ranks = group.get_ranks()
            return set(ranks)

        # Fallback: calculate based on group properties
        current_rank = dist.get_rank()
        local_rank = group.rank()
        group_size = group.size()

        # Calculate base rank (rank 0 of this group)
        base_rank = current_rank - local_rank

        # All ranks in this group
        group_ranks = {base_rank + i for i in range(group_size)}

        return group_ranks

    except Exception as e:
        logger.debug(f"Error getting process group ranks: {e}")
        # Ultimate fallback - just return current rank
        return {dist.get_rank()}


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
    # Get the actual group ranks using the new method
    group_ranks = _get_process_group_ranks(group)
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
