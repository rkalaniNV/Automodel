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
import argparse
import logging
import re
import string
from typing import List

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def _normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace."""

    def _remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def _white_space_fix(text):
        return " ".join(text.split())

    def _remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def _lower(text):
        return text.lower()

    return _white_space_fix(_remove_articles(_remove_punc(_lower(s))))


def _f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 for one prediction and ground-truth pair."""
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens = _normalize_answer(ground_truth).split()

    common = set(pred_tokens) & set(gt_tokens)
    num_same = sum(min(pred_tokens.count(t), gt_tokens.count(t)) for t in common)
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return float(pred_tokens == gt_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def _exact_match(prediction: str, ground_truth: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))


def _metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]) -> float:
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def evaluate(model, tokenizer, dataset, device, max_new_tokens: int = 32, batch_size: int = 1):
    """Iterate over `dataset` and compute SQuAD EM / F1 metrics."""

    em_total = 0.0
    f1_total = 0.0
    num_samples = len(dataset)

    model.eval()

    with torch.no_grad():
        for idx in tqdm(range(0, num_samples, batch_size), desc="Evaluating"):
            batch = dataset[idx : idx + batch_size]

            prompts = [f"{ex['context']} {ex['question']}" for ex in batch]

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            input_len = inputs["input_ids"].shape[1]

            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            preds = tokenizer.batch_decode(generated[:, input_len:], skip_special_tokens=True)

            for pred, ex in zip(preds, batch, strict=False):
                ground_truths = ex["answers"]["text"]
                em_total += _metric_max_over_ground_truths(_exact_match, pred, ground_truths)
                f1_total += _metric_max_over_ground_truths(_f1_score, pred, ground_truths)

    em = 100.0 * em_total / num_samples
    f1 = 100.0 * f1_total / num_samples
    return {"exact_match": em, "f1": f1, "total_samples": num_samples}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a causal-LM checkpoint on the SQuAD dataset.")
    parser.add_argument("--model", type=str, required=True, help="Path or HF Hub ID of the fine-tuned model")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split to evaluate on")
    parser.add_argument("--dataset", type=str, default="rajpurkar/squad", help="HF dataset identifier")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit evaluation to first N samples")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Maximum tokens to generate per sample")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

    logging.info("Loading tokenizer and model from '%s'", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if torch.cuda.is_available() else None
    )
    model.to(args.device)

    logging.info("Loading SQuAD dataset '%s' (%s split)", args.dataset, args.split)
    dataset = load_dataset(args.dataset, split=args.split)
    if args.max_samples:
        dataset = dataset.select(range(args.max_samples))

    metrics = evaluate(model, tokenizer, dataset, args.device, args.max_new_tokens, args.batch_size)
    logging.info(
        "Evaluation complete | EM: %.2f | F1: %.2f | Samples: %d",
        metrics["exact_match"],
        metrics["f1"],
        metrics["total_samples"],
    )


if __name__ == "__main__":
    main()
