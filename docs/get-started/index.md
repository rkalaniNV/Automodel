---
description: "Get started quickly with NeMo Automodel by following these essential setup steps and running your first fine-tuning job."
tags: ["quickstart", "setup", "beginner", "onboarding", "fine-tuning"]
categories: ["getting-started"]
---

(get-started-overview)=
# Get Started with NeMo Automodel

Welcome to NeMo Automodel! This guide will help you set up your environment and run your first fine-tuning job with large language models and vision language models.

## Before You Start

- **System Requirements**: CUDA-compatible GPUs with sufficient memory (8GB+ recommended)
- **Python Environment**: Python 3.10+ with PyTorch 2.0+
- **Model Access**: Access to Hugging Face models (Llama, Gemma, Qwen, etc.)
- **Optional**: Slurm cluster for distributed training at scale

---

## Essential Setup

:::::{grid} 1 1 2 2
:gutter: 2

::::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Installation
:link: installation
:link-type: doc

Install NeMo Automodel and set up your Python environment with all dependencies.

+++
{bdg-success}`Essential`
::::

::::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Quick Start
:link: quick-start
:link-type: doc

Run your first fine-tuning job in minutes with our step-by-step guide.

+++
{bdg-primary}`Beginner`
::::

::::{grid-item-card} {octicon}`search;1.5em;sd-mr-1` Model Selection
:link: model-selection
:link-type: doc

Choose the right model for your task from LLMs, VLMs, and speech models.

+++
{bdg-info}`Planning`
::::

::::{grid-item-card} {octicon}`device-desktop;1.5em;sd-mr-1` Local Workstation
:link: local-workstation
:link-type: doc

Configure your local environment for training on single or multiple GPUs.

+++
{bdg-secondary}`Setup`
::::

:::::

## Choose Your Training Approach

Select the training method that best fits your needs and experience level:

:::::{grid} 1 1 2 2
:gutter: 2

::::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` LLM Fine-tuning
:link: ../guides/llm/sft
:link-type: doc
:link-alt: Language model fine-tuning guide

Supervised fine-tuning for language models. Perfect for beginners and domain adaptation tasks.

+++
{bdg-primary}`Beginner`
::::

::::{grid-item-card} {octicon}`image;1.5em;sd-mr-1` VLM Fine-tuning
:link: ../guides/vlm/index
:link-type: doc
:link-alt: Vision-language model guide

Fine-tune vision language models for multimodal tasks like image captioning and VQA.

+++
{bdg-info}`Multimodal`
::::

::::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` PEFT Training
:link: ../guides/llm/peft
:link-type: doc
:link-alt: Parameter-efficient fine-tuning guide

Memory-efficient LoRA adaptation for large models. Reduce training costs and time.

+++
{bdg-success}`Efficient`
::::

::::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Omni-modal
:link: ../guides/omni/index
:link-type: doc
:link-alt: Omni-modal training guide

Train models on multiple modalities including text, images, and more.

+++
{bdg-warning}`Advanced`
::::

:::::

## Distributed Training

Scale your training across multiple GPUs and nodes:

- **Single Node Multi-GPU**: Use FSDP2 or DDP for training on multiple GPUs
- **Multi-Node Clusters**: Deploy on Slurm clusters with container support
- **Memory Optimization**: Leverage gradient checkpointing and CPU offloading

For details, see the {doc}`../guides/launcher/index` and {doc}`../about/architecture-overview`.

## Next Steps

After completing your first training run:

1. **Choose Your Model**: Use {doc}`model-selection` to pick the right model for your task
2. **Try PEFT**: Experiment with {doc}`../guides/llm/peft` for memory-efficient training
3. **Scale Up**: Set up {doc}`../guides/launcher/slurm` for distributed training
4. **Explore VLMs**: Train vision language models with {doc}`../guides/vlm/index`
5. **Optimize**: Use advanced features like FP8 quantization and kernel optimizations
6. **Troubleshoot**: Check {doc}`../references/troubleshooting-reference` for common issues

## Get Help

- {doc}`../references/troubleshooting-reference` - Comprehensive troubleshooting guide
- {doc}`../api-docs/index` - Complete API documentation  
- {doc}`../references/yaml-configuration-reference` - Configuration parameters
- {doc}`../references/cli-command-reference` - Command-line usage
- {doc}`../about/index` - Platform overview and key features
