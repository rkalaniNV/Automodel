# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from typing import Optional

from nemo_automodel.shared.import_utils import MISSING_TE_PARALLEL_CE_MSG

try:
    from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy

    HAVE_TE_PARALLEL_CE = True
except ImportError:
    HAVE_TE_PARALLEL_CE = False


class TEParallelCrossEntropy:
    def __init__(
        self, 
        ignore_index: int = -100, 
        reduction: str = "sum",
        tp_group: Optional[torch.distributed.ProcessGroup] = None
    ):
        """
        Transformer Engine parallel cross entropy loss.

        Args:
            ignore_index (int): Target value that is ignored when computing the loss. Defaults to -100.
            reduction (str): Type of reduction ('none', 'mean', 'sum'). Defaults to "mean".
            tp_group (Optional[torch.distributed.ProcessGroup]): Process group for tensor parallelism. Defaults to None.
        """
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.tp_group = tp_group

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute parallel cross entropy loss that matches PyTorch's cross_entropy behavior.

        Args:
            logits: Input logits. Shape: [B, T, V]
            labels: Target labels. Shape: [B, T]
            mask: Mask to apply to the loss. Shape: [B, T]

        Returns:
            Computed loss tensor
        """
        if not HAVE_TE_PARALLEL_CE:
            raise ImportError(MISSING_TE_PARALLEL_CE_MSG)

        # Handle ignore_index by replacing ignored labels with a valid index
        # and then masking out those positions
        if self.ignore_index != -100:
            # Create mask for positions to ignore
            ignore_mask = (labels == self.ignore_index)
            # Replace ignore_index with 0 (valid index) for computation
            labels = labels.clone()
            labels[ignore_mask] = 0
            # Update the loss mask to zero out ignored positions
            if mask is not None:
                mask = mask.clone()
                mask[ignore_mask] = 0.0
            else:
                mask = (~ignore_mask).float()
                

        # Compute TE parallel cross entropy
        # Note: TE parallel_cross_entropy expects (logits, labels, label_smoothing, shift, tp_group)
        te_loss = parallel_cross_entropy(logits, labels, 0.0, False, self.tp_group)

        # Apply loss mask if provided
        if mask is not None:
            te_loss = te_loss * mask.float()

        # Apply reduction
        if self.reduction == "none":
            return te_loss
        elif self.reduction == "mean":
            if mask is not None:
                # Mean over valid (non-masked) positions
                return te_loss.sum() / mask.sum().clamp(min=1.0)
            else:
                return te_loss.mean()
        elif self.reduction == "sum":
            return te_loss.sum()
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}. Must be one of 'none', 'mean', 'sum'")
