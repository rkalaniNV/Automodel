from contextlib import ExitStack
from typing import TYPE_CHECKING

import torch
from torch.utils.data import IterableDataset

from nemo_automodel.components.datasets.llm.lingua_assets.core import (
    build_dataloader_from_args,
    init_dataloader_state_from_args,
)

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class JSONLDataset(IterableDataset):
    def __init__(
        self,
        root_dir: str,
        rank: int,
        world_size: int,
        tokenizer: "PreTrainedTokenizerBase",
        sources: dict[str, float],
        batch_size: int,
        packed_sequence_size: int,
        seed: int,
        split: str,
        add_bos: bool = True,
        add_eos: bool = True,
        load_async: bool = False,
        prefetch_size: int = 64,
        n_views: int = 2,
    ):
        assert split in ["train", "validation"], "Split must be either train or validation"
        # Persist constructor args so we can rebuild the dataloader after a checkpoint restore
        self._root_dir = root_dir
        self._rank = rank
        self._world_size = world_size
        self._tokenizer = tokenizer
        self._sources = sources
        self._batch_size = batch_size
        self._packed_sequence_size = packed_sequence_size
        self._seed = seed
        self._split = split
        self._add_bos = add_bos
        self._add_eos = add_eos
        self._load_async = load_async
        self._prefetch_size = prefetch_size
        self._n_views = n_views

        # Initialize dataloader state
        self.data_loader_state = init_dataloader_state_from_args(
            root_dir,
            rank,
            world_size,
            sources,
            batch_size,
            packed_sequence_size,
            seed,
            add_bos,
            add_eos,
            prefetch_size,
            n_views,
            split,
        )

        # Create the context stack to manage the dataloader lifecycle and build the dataloader
        self._build_dataloader()

    def __iter__(self):
        for batch, state in self.data_loader:
            self.data_loader_state = state

            batch = torch.from_numpy(batch)
            bs, s, _ = batch.shape
            loss_mask = torch.ones((bs, s), dtype=torch.int64)
            return_batch = {
                "input_ids": batch[:, :, 0],
                "labels": batch[:, :, 1],
                "loss_mask": loss_mask,
            }
            yield return_batch

    def __del__(self):
        # Ensure resources are closed even if build failed partially
        if hasattr(self, "context_stack"):
            self.context_stack.close()

    def state_dict(self):
        return self.data_loader_state

    def load_state_dict(self, state_dict):
        """Restore dataloader state from a checkpoint.

        We recreate the underlying dataloader so that it picks up the new state
        (the original dataloader captured the old state at construction time).
        """
        # Close the existing dataloader to release file handles / worker processes
        if hasattr(self, "context_stack"):
            self.context_stack.close()

        # Replace state dict and rebuild the dataloader
        self.data_loader_state = state_dict
        self._build_dataloader()

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _build_dataloader(self):
        """Helper to (re)build the internal dataloader using the current state."""
        # Always create a fresh context stack so resources are correctly managed
        self.context_stack = ExitStack()
        self.data_loader = self.context_stack.enter_context(
            build_dataloader_from_args(
                self._tokenizer, self._load_async, self._prefetch_size, state=self.data_loader_state
            )
        )
