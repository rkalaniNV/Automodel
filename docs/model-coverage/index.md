# About NeMo Automodel Model Coverage

Discover which model architectures and checkpoints you can immediately fine-tune and train with NeMo Automodel.

## Overview

NeMo Automodel provides Day-0 support for models from the Hugging Face Hub, enabling immediate fine-tuning without conversion or special setup. Our compatibility spans across multiple model families and architectures, ensuring you can work with the latest models as soon as they're released.

## Supported Model Types

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`comment-discussion;1.5em;sd-mr-1` Large Language Models
:link: llm
:link-type: doc
:link-alt: LLM model coverage

Comprehensive support for text generation models including Llama, Mistral, Gemma, Qwen, and many more architectures.
+++
{bdg-primary}`LLMs`
{bdg-secondary}`Text Generation`
:::

:::{grid-item-card} {octicon}`image;1.5em;sd-mr-1` Vision Language Models
:link: vlm
:link-type: doc
:link-alt: VLM model coverage

Support for multimodal models that process both text and images for VQA, captioning, and conversation tasks.
+++
{bdg-info}`VLMs`
{bdg-secondary}`Multimodal`
:::

::::

## Key Features

- **Day-0 Compatibility**: Use any compatible Hugging Face model immediately
- **No Conversion Required**: Direct integration with Hugging Face checkpoints
- **Automatic Detection**: Intelligent model architecture detection and setup
- **Flexible Training**: Support for both SFT and PEFT across all model types
- **Optimized Performance**: Built-in optimizations for supported architectures

## Architecture Support

NeMo Automodel supports a wide range of model architectures:

### Popular Families
- **Llama** (Llama 2, Llama 3.1, Llama 3.2)
- **Mistral** (Mistral 7B, Mixtral)
- **Gemma** (Gemma 2, Gemma 3n)
- **Qwen** (Qwen 2.5, Qwen VL)
- **DeepSeek** (DeepSeek R1, DeepSeek V3)

### Enterprise Models
- **NVIDIA Nemotron** models
- **IBM Granite** series
- **Microsoft Phi** models

## Testing and Validation

All supported models undergo comprehensive testing with:
- **FSDP2 Distributed Training**: Multi-GPU and multi-node scaling
- **SFT and PEFT**: Both full and parameter-efficient fine-tuning
- **Memory Optimization**: FP8 quantization and gradient checkpointing
- **Performance Benchmarking**: Training speed and convergence validation

## Get Started

1. **Choose your model** from the Hugging Face Hub
2. **Verify compatibility** using our model lists
3. **Set up your configuration** with appropriate settings
4. **Start training** using NeMo Automodel recipes

```{toctree}
:maxdepth: 2
:hidden:

llm
vlm
```
