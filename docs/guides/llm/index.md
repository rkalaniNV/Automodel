---
description: "Learn how to fine-tune large language models using NeMo Automodel with full parameter training and parameter-efficient techniques."
categories: ["model-training"]
tags: ["llm", "fine-tuning", "peft", "lora", "yaml-config", "recipes"]
personas: ["mle-focused", "researcher-focused", "data-scientist-focused"]
difficulty: "intermediate"
content_type: "concept"
modality: "llm"
---

# Train Large Language Models

Learn how to fine-tune Large Language Models (LLMs) using NeMo Automodel with both full parameter training and parameter-efficient techniques.

## Training Methods

NeMo Automodel provides flexible approaches for fine-tuning LLMs to meet your specific requirements and computational constraints.

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Supervised Fine-Tuning (SFT)
:link: sft
:link-type: doc
:link-alt: Full parameter fine-tuning guide

Complete guide to full-parameter fine-tuning for deep model adaptation. Learn when and how to use SFT for maximum performance.
+++
{bdg-primary}`SFT`
{bdg-secondary}`Full Training`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Parameter-Efficient Fine-Tuning (PEFT)
:link: peft
:link-type: doc
:link-alt: PEFT and LoRA training guide

Efficient fine-tuning using LoRA and other PEFT techniques. Achieve great results with minimal computational resources.
+++
{bdg-info}`PEFT`
{bdg-secondary}`LoRA`
:::

::::

## Data Preparation

Learn how to prepare and work with datasets for LLM training, including custom dataset integration and preprocessing.

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Dataset Integration
:link: dataset
:link-type: doc
:link-alt: Dataset preparation guide

Learn how to integrate custom datasets and prepare data for training with various formats and preprocessing techniques.
+++
{bdg-success}`Datasets`
{bdg-secondary}`Custom Data`
:::

:::{grid-item-card} {octicon}`list-unordered;1.5em;sd-mr-1` Instruction Datasets
:link: column-mapped-text-instruction-dataset
:link-type: doc
:link-alt: Instruction dataset guide

Working with instruction-following datasets and conversation formats for chat and instruction-tuned models.
+++
{bdg-warning}`Instructions`
{bdg-secondary}`Chat Format`
:::

::::

## Key Features

- **Day-0 Support**: Use any Hugging Face model immediately without conversion
- **Flexible Training**: Choose between SFT and PEFT based on your needs
- **Optimized Performance**: Benefit from BF16/FP8 quantization and kernel optimizations
- **Distributed Training**: Scale from single GPU to multi-node clusters with FSDP2/nvFSDP
- **Easy Deployment**: Export to vLLM, TensorRT-LLM, and other inference frameworks

```{toctree}
:maxdepth: 2
:hidden:

sft
peft
dataset
column-mapped-text-instruction-dataset
```
