from datasets import load_dataset
def _max_len(ds, block_size=None, make_mult_of_8=False):
    max_len = max(map(lambda x: len(x["input_ids"]), ds))
    # multiple of 8
    if make_mult_of_8:
        max_len = ((max_len // 8) + 1) * 8
    if block_size is not None:
        max_len = min(max_len, block_size)
    return max_len

# def _pad_fn(self, max_len):
#     pad_id = self.tokenizer.pad_token_id or 0

def _pad(batch, pad_id, max_len):
    batch["input_ids"] = [
        ids + [pad_id] * (max_len - len(ids)) for ids in batch["input_ids"]
    ]
    batch["attention_mask"] = [
        [1] * len(ids) + [0] * (max_len - len(ids))
        for ids in batch["attention_mask"]
    ]
    batch["labels"] = [
        label + [-100] * (max_len - len(label))
        for label in batch["labels"]
    ]
    # safety truncate
    return {k: v[:max_len] if isinstance(v, list) else v for k, v in batch.items()}

class HellaSwag:
    """
    HellaSwag single-turn supervised-fine-tuning (SFT) dataset.

    - Loads a split of HellaSwag (optionally sliced with num_samples_limit).
    - Tokenises context + correct ending.
    - Builds labels where context tokens are masked with -100.
    - Pads every sample to a global max-length (rounded to nearest multiple of 8).
    """

    # ------------------------------------------------------------------ #
    # construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        path_or_dataset: str,
        tokenizer,
        split: str = "train",
        num_samples_limit: int | None = None,
        trust_remote_code: bool = True,
        block_size: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.preprocessing_num_workers = 1
        self.overwrite_cache = False

        # slice split if requested
        if isinstance(num_samples_limit, int):
            split = f"{split}[:{num_samples_limit}]"

        raw_ds = load_dataset(path_or_dataset, split=split,
                              trust_remote_code=trust_remote_code)

        # ensure pad token exists
        if not getattr(self.tokenizer, "pad_token", None) \
           and getattr(self.tokenizer, "bos_token", None):
            self.tokenizer.pad_token = self.tokenizer.bos_token

        # 1) tokenise ----------------------------------------------------
        self.dataset = raw_ds.map(
            lambda ex: self._tokenize(ex),
            batched=True,
            num_proc=self.preprocessing_num_workers,
            remove_columns=raw_ds.column_names,
            load_from_cache_file=False,
            desc="Tokenizing HellaSwag",
        )

        # 2) compute global max length -----------------------------------
        # max_len = _max_len(tokenised, block_size)

        # 3) pad ----------------------------------------------------------
        # pad_fn = self._pad_fn(max_len)
        # pad_id = self.tokenizer.pad_token_id or 0
        # self.dataset = tokenised
        # self.dataset = tokenised.map(
        #     lambda x: _pad(x, pad_id, max_len),
        #     batched=True,
        #     num_proc=self.preprocessing_num_workers,
        #     load_from_cache_file=not self.overwrite_cache,
        #     desc=f"Padding to max length {max_len}",
        # )

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_context(examples):
        return examples["ctx"]

    @staticmethod
    def _get_target(examples):
        # Choose the gold ending according to the integer label.
        return [
            endings[int(lbl)] for endings, lbl in zip(examples["endings"], examples["label"])
        ]

    def _tokenize(self, examples):
        ctx = self._get_context(examples)
        tgt = self._get_target(examples)

        ctx_tok = self.tokenizer(ctx)
        tgt_tok = self.tokenizer(tgt)

        # remove trailing special token from context
        if ctx_tok["input_ids"][0] \
           and ctx_tok["input_ids"][0][-1] in self.tokenizer.all_special_ids:
            ctx_tok["input_ids"]       = [ids[:-1] for ids in ctx_tok["input_ids"]]
            ctx_tok["attention_mask"]  = [msk[:-1] for msk in ctx_tok["attention_mask"]]

        # remove leading special token from target
        if tgt_tok["input_ids"][0] \
           and tgt_tok["input_ids"][0][0] in self.tokenizer.all_special_ids:
            tgt_tok["input_ids"]      = [ids[1:] for ids in tgt_tok["input_ids"]]
            tgt_tok["attention_mask"] = [msk[1:] for msk in tgt_tok["attention_mask"]]

        # concat
        input_ids      = [c + t for c, t in zip(ctx_tok["input_ids"],      tgt_tok["input_ids"])]
        attention_mask = [c + t for c, t in zip(ctx_tok["attention_mask"], tgt_tok["attention_mask"])]

        # labels: context -> -100, target -> real ids, plus -100 after EOS
        labels = [
            [-100] * (len(ctx) - 1) + tgt + [-100]
            for ctx, tgt in zip(ctx_tok["input_ids"], tgt_tok["input_ids"])
        ]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    # ------------------------------------------------------------------ #
    # dataset-like interface                                             #
    # ------------------------------------------------------------------ #
    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
