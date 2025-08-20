---
description: "Plug-and-play training recipes for LLM and VLM fine-tuning using NeMo Automodel"
categories: ["learning-resources"]
tags: ["recipes", "llm", "vlm", "lora", "fsdp", "training"]
personas: ["mle-focused", "researcher-focused", "data-scientist-focused"]
difficulty: "beginner"
content_type: "how-to"
modality: "universal"
---

# Training Recipes (LLM & VLM)

NeMo Automodel ships with plug-and-play training scripts ("recipes") that let you fine-tune popular models with minimal setup. Pick a recipe, choose a YAML config, and run.

## What You Can Do

- Fine-tune LLMs such as LLaMA, Qwen, Gemma, Phi, and more
- Train VLMs such as Gemma-3-VL and Qwen2.5-VL
- Use advanced techniques including LoRA (PEFT) and distributed training with FSDP/nvFSDP

## Quick Start

Run an LLM fine-tuning recipe with uv:

```bash
uv run recipes/llm/finetune.py --config recipes/llm/llama_3_2_1b_hellaswag.yaml
```

Adjust the config path to match your target model and dataset.

## LLM Recipes

LLM recipes provide out-of-the-box fine-tuning flows for next‑token prediction tasks with options for PEFT, FP8, packed sequences, and distributed strategies.

- Entry point: `nemo_automodel/recipes/llm/finetune.py`
- Example configs: see the repository under `examples/llm/`

Capabilities:
- Supervised fine-tuning (SFT)
- LoRA/DoRA via PEFT
- FSDP2/nvFSDP for multi-GPU and multi-node
- Packed sequence training and mixed precision

## VLM Recipes

VLM recipes support multimodal fine-tuning with image‑text inputs using model‑appropriate processors and collate functions.

- Entry point: `nemo_automodel/recipes/vlm/finetune.py`
- Example configs: see the repository under `examples/vlm/`

Capabilities:
- Vision-language fine-tuning with LoRA
- Image/text processors via Hugging Face AutoProcessor
- FSDP2/nvFSDP for scalable training

## Where to Find Recipes

- Python scripts (entry points):
  - `nemo_automodel/recipes/llm/finetune.py`
  - `nemo_automodel/recipes/vlm/finetune.py`

- Example YAML configs to copy/modify:
  - `examples/llm/`
  - `examples/vlm/`

## See Also

- {doc}`../get-started/installation` – install and environment setup
- {doc}`../guides/llm/index` – language model guides
- {doc}`../guides/vlm/index` – vision‑language model guides

