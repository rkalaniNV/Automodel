---
description: "Learn how to fine-tune omni-modal models like Gemma 3n that support multiple input modalities including text, images, and audio."
tags: ["omni", "multimodal", "gemma", "gemma3n"]
categories: ["models"]
---

# Train Omni-Modal Models

Learn how to fine-tune advanced omni-modal models that can process multiple input types including text, images, audio, and more using NeMo Automodel.

## Overview

Omni-modal models represent the next generation of AI systems that can seamlessly understand and reason across multiple modalities. These models can process diverse inputs like text, images, audio, and video within a single unified architecture, enabling more natural and versatile AI interactions.

## Supported Models

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Gemma 3n
:link: gemma3-3n
:link-type: doc
:link-alt: Gemma 3n fine-tuning guide

Fine-tune Google's Gemma 3n model with optimized architecture featuring MatFormer and efficient resource usage for multimodal tasks.
+++
{bdg-primary}`Gemma 3n`
{bdg-secondary}`Google`
:::

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` More Models Coming
:link: gemma3-3n
:link-type: doc
:link-alt: Future omni-modal models

Support for additional omni-modal architectures will be added in future releases.
+++
{bdg-info}`Coming Soon`
{bdg-secondary}`Future`
:::

::::

## Key Capabilities

- **Multi-Modal Input**: Process text, images, audio, and other modalities simultaneously
- **Unified Architecture**: Single model handles diverse input types without separate encoders
- **Optimized Performance**: Benefit from advanced architectures like MatFormer and Per-Layer Embeddings
- **Efficient Training**: Support for both full fine-tuning and PEFT techniques
- **Flexible Deployment**: Export to various inference frameworks for production use

## Gemma 3n Features

Gemma 3n introduces several innovations:

- **MatFormer Architecture**: Nested transformers for improved efficiency
- **Per-Layer Embedding Caching**: Reduced memory usage and faster inference
- **Sub-model Extraction**: Extract smaller models from larger trained models
- **KV Cache Sharing**: Optimized memory usage during inference
- **Multi-Modal Integration**: Built-in image and audio encoders

## Training Options

- **Supervised Fine-Tuning (SFT)**: Full parameter updates for maximum adaptation
- **Parameter-Efficient Fine-Tuning (PEFT)**: LoRA and other efficient techniques
- **Mixed Training**: Combine different modalities in training datasets
- **Distributed Training**: Scale across multiple GPUs with FSDP2/nvFSDP

```{toctree}
:maxdepth: 2
:hidden:

gemma3-3n
```