from typing import TYPE_CHECKING
from torch.utils.data import IterableDataset
from nemo_automodel.components.datasets.llm.lingua_assets.core import build_dataloader_from_args, init_dataloader_state_from_args
from contextlib import ExitStack
import torch

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

class JSONLDataset(IterableDataset):
    def __init__(self, root_dir: str, rank: int, world_size: int, tokenizer: "PreTrainedTokenizerBase", sources: dict[str, float], batch_size: int, packed_sequence_size: int, seed: int, split: str, add_bos: bool = True, add_eos: bool = True, load_async: bool = False, prefetch_size: int = 64, n_views: int = 2):
        assert split in ["train", "validation"], "Split must be either train or validation"
        # Initialize dataloader state
        self.data_loader_state = init_dataloader_state_from_args(
            root_dir, rank, world_size, sources, batch_size, packed_sequence_size, seed, add_bos, add_eos, prefetch_size, n_views, split
        )
        
        # Create the context stack to manage the dataloader lifecycle
        self.context_stack = ExitStack()
        
        # Build the dataloader within the context
        self.data_loader = self.context_stack.enter_context(
            build_dataloader_from_args(tokenizer, load_async, prefetch_size, state=self.data_loader_state)
        )

    def __iter__(self):
        for batch, state in self.data_loader:
            batch = torch.from_numpy(batch)
            bs, s, _ = batch.shape
            loss_mask = torch.Tensor([1 for _ in range(s)]).unsqueeze(0).repeat(bs, 1).to(torch.int64)
            return_batch = {
                "input_ids": batch[:, :, 0],
                "labels": batch[:, :, 1],
                "loss_mask": loss_mask,
            }
            yield return_batch
            self.data_loader_state = state
    
    def __del__(self):
        self.context_stack.close()
    
    def state_dict(self):
        return self.data_loader_state
    
    def load_state_dict(self, state_dict):
        self.data_loader_state = state_dict
