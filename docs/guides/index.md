---
description: "Comprehensive guides for training language models, vision language models, and omni-modal models with NeMo Automodel across different environments and configurations."
tags: ["training", "guides", "llm", "vlm", "omni", "checkpointing", "launcher"]
categories: ["training", "deployment"]
---

# About NeMo Automodel Training Guides

Master fine-tuning and deployment workflows with comprehensive guides for language models, vision language models, and advanced omni-modal architectures.

## Training by Model Type

Choose your training approach based on the type of model and task you want to accomplish.

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} {octicon}`comment-discussion;1.5em;sd-mr-1` Language Models
:link: llm/index
:link-type: doc
:link-alt: LLM training guides

**Complete LLM fine-tuning workflows**

- Supervised Fine-Tuning (SFT)
- Parameter-Efficient Fine-Tuning (PEFT)
- Custom dataset integration
- Instruction-following models

+++
{bdg-primary}`LLM`
{bdg-secondary}`Text`
:::

:::{grid-item-card} {octicon}`image;1.5em;sd-mr-1` Vision Language Models
:link: vlm/index
:link-type: doc
:link-alt: VLM training guides

**Multi-modal model training**

- Visual question answering
- Image captioning
- Multi-modal conversations
- Medical and scientific VQA

+++
{bdg-info}`VLM`
{bdg-secondary}`Multi-modal`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Omni-Modal Models
:link: omni/index
:link-type: doc
:link-alt: Omni-modal training guides

**Advanced multi-modal architectures**

- Gemma 3n with MatFormer
- Text, image, audio processing
- Sub-model extraction
- Per-layer embedding caching

+++
{bdg-warning}`Omni`
{bdg-secondary}`Advanced`
:::

::::

## Infrastructure & Deployment

Learn how to deploy training jobs across different computing environments and manage training checkpoints.

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Job Launcher
:link: launcher/index
:link-type: doc
:link-alt: Job launcher guides

**Multi-environment deployment**

- SLURM cluster integration
- Local workstation setup
- Distributed training coordination
- Container support

+++
{bdg-success}`Launcher`
{bdg-secondary}`Scaling`
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Checkpointing
:link: checkpointing
:link-type: doc
:link-alt: Checkpointing guide

**Training state management**

- Safetensors format support
- PyTorch DCP checkpointing
- Hugging Face compatibility
- Resume training workflows

+++
{bdg-info}`Checkpoints`
{bdg-secondary}`Storage`
:::

::::

## Key Features Across All Guides

- **Day-0 Hugging Face Support**: Use any compatible model without conversion
- **Optimized Performance**: BF16/FP8 quantization and kernel optimizations
- **Flexible Scaling**: Single GPU to multi-node distributed training with FSDP2/nvFSDP
- **Easy Deployment**: Export to vLLM, TensorRT-LLM, and other inference frameworks
- **Production Ready**: Safetensors checkpoints compatible with Hugging Face ecosystem

## Training Workflow Overview

1. **Choose Your Model Type**: Start with {doc}`llm/index`, {doc}`vlm/index`, or {doc}`omni/index`
2. **Prepare Your Environment**: Set up using {doc}`launcher/index` for your infrastructure
3. **Configure Training**: Select between SFT and PEFT based on your requirements
4. **Monitor & Checkpoint**: Use {doc}`checkpointing` for reliable training state management
5. **Deploy**: Export models for inference with full Hugging Face compatibility

## Getting Started

New to NeMo Automodel? Start with these essential guides:

1. **{doc}`llm/sft`** - Basic supervised fine-tuning workflow
2. **{doc}`llm/peft`** - Memory-efficient training with LoRA
3. **{doc}`launcher/slurm`** - Scale to multi-node clusters
4. **{doc}`checkpointing`** - Understand checkpoint formats and resuming
