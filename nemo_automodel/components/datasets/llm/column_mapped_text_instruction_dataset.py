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

import logging
import re
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Union

from datasets import load_dataset
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

# Supported cases:
# Format:
# - Context + question + answer
# - Question + answer
# Input types:
# - one or more paths to jsonl files
# - dataset id from huggingface.


class ColumnTypes(Enum):
    Context = "context"
    Question = "question"
    Answer = "answer"


def make_iterable(val: Union[str, List[str]]) -> Iterator[str]:
    """Utility that converts *val* into an iterator of strings.

    The helper accepts either a single string or a list of strings and
    yields its contents. This is handy when we want to treat the two cases
    uniformly downstream (e.g. when iterating over *data_files* that can be
    provided as either a single path or a collection of paths).

    Args:
        val: Either a single string or a list/tuple of strings.

    Yields:
        str: The individual strings contained in *val*.

    Raises:
        ValueError: If *val* is neither a string nor an iterable of strings.
    """
    if isinstance(val, str):
        yield val
    elif isinstance(val, (list, tuple)):
        for item in val:
            if not isinstance(item, str):
                raise ValueError("All elements must be strings")
            yield item
    else:
        raise ValueError(f"Expected str or list[str], got {type(val)}")


def _str_is_hf_repo_id(val: str) -> bool:
    """
    Check if a string is a valid huggingface dataset id.

    Args:
        val: A string to check.

    Returns:
        True if the string is a valid huggingface dataset id, False otherwise.
    """
    return re.match(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$", val) is not None and not Path(val).exists()


def _load_dataset(path_or_dataset_id: Union[str, List[str]], split: Optional[str] = None, streaming: bool = False):
    """Load a dataset either from the Hugging Face Hub or from local JSON/JSONL files.

    If *path_or_dataset_id* resembles a HF repo ID (i.e. of the form
    ``org/dataset`` and the path does **not** exist on the local filesystem),
    we defer to ``datasets.load_dataset`` directly. Otherwise, we assume the
    argument points to one or more local JSON/JSONL files and let
    ``datasets.load_dataset`` with the *"json"* script handle the parsing.

    Args:
        path_or_dataset_id: Either a HF dataset identifier (``org/name``) or
            a path / list of paths to local ``.json`` / ``.jsonl`` files.
        split: Optional split to load when retrieving a remote dataset. This
            parameter is ignored for local files as the *json* script always
            returns a single split.

    Returns:
        datasets.Dataset: The loaded dataset.
    """
    if isinstance(path_or_dataset_id, str) and _str_is_hf_repo_id(path_or_dataset_id):
        return load_dataset(path_or_dataset_id, split=split or "train", streaming=streaming)

    data_files = list(make_iterable(path_or_dataset_id))
    if not data_files:
        raise RuntimeError("No data files provided")

    return load_dataset("json", data_files=data_files, split="train", streaming=streaming)


def _apply_tokenizer_with_chat_template(
    tokenizer: "PreTrainedTokenizer",
    context: str,
    question: str,
    answer: str,
    start_of_turn_token: Optional[str] = None,
    answer_only_loss_mask: bool = True,
) -> Dict[str, List[int]]:
    """
    Tokenization path when the tokenizer supports a chat template.

    Args:
        context: The context of the sample.
        question: The question of the sample.
        answer: The answer of the sample.

    Returns:
        A dictionary with the tokenized columns.
    """

    # Build conversation
    user_prompt = "".join(filter(None, [context, " " if context and question else "", question])).strip()
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": answer},
    ]

    input_ids: List[int] = tokenizer.apply_chat_template(messages)


    # Loss mask computation
    loss_mask: Optional[List[int]] = None
    if answer_only_loss_mask:
        if isinstance(start_of_turn_token, str):
            start_ids = tokenizer(start_of_turn_token, add_special_tokens=False)["input_ids"]
            start_id = start_ids[0]
            first_idx = input_ids.index(start_id) if start_id in input_ids else 0
            if input_ids.count(start_id) >= 2:
                response_start = input_ids.index(start_id, first_idx + 1)
            else:
                response_start = 0
        else:
            response_start = 0
        loss_mask = [0] * response_start + [1] * (len(input_ids) - response_start)

    # Build labels and apply loss mask directly
    labels = input_ids[1:].copy()
    input_ids = input_ids[:-1]

    if loss_mask is not None:
        loss_mask = loss_mask[1:]  # Shift together with labels
        # Apply loss_mask directly to labels: set ignored positions to -100
        labels = [-100 if mask == 0 else label for label, mask in zip(labels, loss_mask)]

    out: Dict[str, List[int]] = {
        "input_ids": input_ids,
        "labels": labels,
    }
    return out


def _apply_tokenizer_plain(
    tokenizer: "PreTrainedTokenizer", context: str, question: str, answer: str, answer_only_loss_mask: bool = True
) -> Dict[str, List[int]]:
    """
    Tokenization path when *chat_template* is not available.

    Args:
        context: The context of the sample.
        question: The question of the sample.
        answer: The answer of the sample.

    Returns:
        A dictionary with the tokenized columns.
    """

    if context and question:
        prompt_text = f"{context} {question} "
    elif context:
        prompt_text = f"{context} "
    elif question:
        prompt_text = f"{question} "
    else:
        raise ValueError("Context and question are both missing")

    # Tokenize
    eos_id = getattr(tokenizer, "eos_token_id", None)
    bos_id = getattr(tokenizer, "bos_token_id", None)

    prompt_ids: List[int] = tokenizer(prompt_text)["input_ids"]
    answer_ids: List[int] = tokenizer(str(answer).strip())["input_ids"]

    # Strip trailing EOS from prompt and leading BOS from answer
    if prompt_ids and eos_id is not None and prompt_ids[-1] == eos_id:
        prompt_ids = prompt_ids[:-1]
    if answer_ids and bos_id is not None and answer_ids[0] == bos_id:
        answer_ids = answer_ids[1:]

    input_ids = prompt_ids + answer_ids
    labels = input_ids[1:]
    input_ids = input_ids[:-1]

    # Apply loss mask directly to labels
    if answer_only_loss_mask:
        loss_mask = [0] * (len(prompt_ids) - 1) + [1] * len(answer_ids)
        # Apply loss_mask directly to labels: set ignored positions to -100
        labels = [-100 if mask == 0 else label for label, mask in zip(labels, loss_mask)]

    out: Dict[str, List[int]] = {
        "input_ids": input_ids,
        "labels": labels,
    }
    return out


def _has_chat_template(tokenizer: "PreTrainedTokenizer") -> bool:
    """
    Check if the tokenizer supports a chat template.

    Args:
        tokenizer: The tokenizer to check.

    Returns:
        True if the tokenizer supports a chat template, False otherwise.
    """
    return getattr(tokenizer, "chat_template", None) is not None and callable(
        getattr(tokenizer, "apply_chat_template", None)
    )


def _check_all_values_equal_length(sample: Dict[str, List[int]]) -> bool:
    """
    Check if all values in the sample are of the same length.
    """
    len0 = len(sample[next(iter(sample))])
    return all(map(lambda v: len(v) == len0, sample.values()))


class ColumnMappedTextInstructionDataset(Dataset):
    """Generic *instruction‐tuning* dataset that maps arbitrary column names.

    The class is intentionally lightweight: it simply loads the raw samples
    (either from HF or from local JSON/JSONL files) and remaps the columns so
    that downstream components can rely on a consistent field interface.

    Optionally, if *answer_only_loss_mask* is requested, the dataset will also
    compute a *loss_mask* indicating which tokens should contribute to the
    loss (typically only those belonging to the assistant answer).
    """

    def __init__(
        self,
        path_or_dataset_id: Union[str, List[str]],
        column_mapping: Dict[str, str],
        tokenizer,
        *,
        split: Optional[str] = None,
        streaming: bool = False,
        answer_only_loss_mask: bool = True,
        start_of_turn_token: Optional[str] = None,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            path_or_dataset_id: The path or dataset id of the dataset.
            column_mapping: The mapping of the columns.
            tokenizer: The tokenizer to use.
            split: The split of the dataset to load.
            streaming: Whether to load the dataset in streaming mode.
            answer_only_loss_mask: Whether to compute the loss mask only on the answer tokens.
            start_of_turn_token: The token to use to indicate the start of a turn.
        """

        if _has_chat_template(tokenizer):
            if not answer_only_loss_mask:
                logging.warning(
                    "answer_only_loss_mask=False but tokenizer has chat template. Consider providing `answer_only_loss_mask` and `start_of_turn_token`."
                )
            elif start_of_turn_token is None:
                raise ValueError("start_of_turn_token must be provided when answer_only_loss_mask=True")

        assert tokenizer is not None, "Tokenizer is required"
        self.tokenizer = tokenizer

        self.streaming = streaming
        self.dataset = _load_dataset(path_or_dataset_id, split=split, streaming=streaming)

        # Keep mapping: dest -> source (i.e. public_field -> raw_column_name)

        assert isinstance(column_mapping, dict), "Expected column_mapping to be a dictionary"
        # Ensure required columns are present
        assert ColumnTypes.Question.value in column_mapping, (
            "Expected question to be in column_mapping",
            column_mapping,
        )
        assert ColumnTypes.Answer.value in column_mapping, ("Expected answer to be in column_mapping", column_mapping)
        if len(column_mapping) == 3:
            assert ColumnTypes.Context.value in column_mapping, (
                "Expected context to be in column_mapping",
                column_mapping,
            )

        self.column_mapping = column_mapping

        self.answer_only_loss_mask = answer_only_loss_mask
        self.start_of_turn_token = start_of_turn_token

    def __len__(self) -> int:  # noqa: D401
        """
        Returns the length of the dataset.

        Returns:
            The length of the dataset.

        Raises:
            RuntimeError: If streaming is enabled.
        """
        if self.streaming:
            raise RuntimeError("Streaming datasets do not have a defined length")
        return len(self.dataset)

    def __getitem__(self, idx):  # noqa: D401
        """
        Returns the item at the given index.

        Args:
            idx: The index of the item to return.

        Returns:
            A dictionary with the mapped columns.

        Raises:
            RuntimeError: If streaming is enabled.
        """
        if self.streaming:
            raise RuntimeError("__getitem__ is not supported when `streaming=True`. Iterate over the dataset instead.")
        row = self.dataset[idx]
        mapped = {dest: row[src] for dest, src in self.column_mapping.items()}
        mapped = self._apply_tokenizer(mapped)
        assert _check_all_values_equal_length(mapped), "All values must be of the same length"
        return mapped

    def __iter__(self):  # noqa: D401
        """
        Iterate over the dataset yielding rows with the requested column mapping.

        When *streaming=True* the underlying dataset is consumed lazily.

        If the tokenizer is provided, it will be used to tokenize the dataset.

        Returns:
            An iterator over the dataset.

        Raises:
            RuntimeError: If streaming is enabled.
        """
        if self.streaming:
            for row in self.dataset:
                mapped = {dest: row[src] for dest, src in self.column_mapping.items()}
                mapped = self._apply_tokenizer(mapped)
                assert _check_all_values_equal_length(mapped), "All values must be of the same length"
                yield mapped
        else:
            for idx in range(len(self)):
                # Reuse __getitem__ to avoid duplicating logic.
                yield self[idx]

    def _apply_tokenizer(self, sample: Dict[str, str]) -> Dict[str, List[int]]:
        """
        Tokenize a mapped *sample* and compute auxiliary fields.

        If the tokenizer is provided:
        - If the tokenizer supports a chat template, the dataset will be tokenized in a conversation style.
        - Otherwise, the dataset will be tokenized in a simple prompt-completion style.

        Args:
            sample: A dictionary with the mapped columns.

        Returns:
            A dictionary with the tokenized columns.
        """
        assert isinstance(sample, dict), "Expected sample to be a dictionary"
        assert len(sample) >= 2, "Expected at least two columns"
        context = sample.get(ColumnTypes.Context.value, None)
        question = sample[ColumnTypes.Question.value]
        answer = sample[ColumnTypes.Answer.value]

        if _has_chat_template(self.tokenizer):
            return _apply_tokenizer_with_chat_template(
                self.tokenizer, context, question, answer, self.start_of_turn_token, self.answer_only_loss_mask
            )

        return _apply_tokenizer_plain(self.tokenizer, context, question, answer, self.answer_only_loss_mask)
