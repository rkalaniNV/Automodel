---
description: "Learn about our platform's core concepts, key features, and fundamental architecture to understand how it works."
tags: ["overview", "concepts", "architecture", "features"]
categories: ["concepts"]
---

(about-overview)=
# About NeMo Automodel

NVIDIA NeMo Automodel is a powerful feature within the NVIDIA NeMo framework that streamlines the process of working with pre-trained models from the Hugging Face Hub. It offers a high-level, user-friendly interface for developers while providing a robust, technically advanced backend for accelerated and scalable training.

## What is Automodel?

Automodel provides Day-0 support for Hugging Face models inside the NeMo framework, so you can use new models immediately—no conversions or rewrites. It bridges HF’s breadth with NeMo’s performance and scaling. Key capabilities include Day-0 Hugging Face integration, accelerated training with BF16/FP8, distributed scaling via DDP and FSDP2, PEFT/LoRA for efficient fine-tuning, and YAML-driven recipes.

## Core Architecture and Technical Details

Automodel integrates Hugging Face architectures into NeMo training pipelines via a dynamic, schema-based configuration layer and optimized runtime primitives:

- **Entry Points**: Drop-in classes such as `NeMoAutoModelForCausalLM` and `NeMoAutoModelForImageTextToText` wrap `transformers` auto-classes, detect model types, and instantiate them with NeMo-compatible wrappers. The wrappers preserve the public API while enabling NeMo-specific optimizations.
- **Performance Optimizations**:
  - SDPA attention paths with preferred backends and optional Liger kernels for faster attention operations.
  - Optional FP8 quantization support for accelerated training on supported hardware, alongside mixed precision (e.g., BF16).
- **Memory and Scale**:
  - NeMo-native FSDP2 sharding strategies to distribute parameters, gradients, and optimizer states, enabling training of very large models.
  - Works with standard DDP for simpler multi-GPU setups.
- **PEFT Integration**: Built-in LoRA enables efficient fine-tuning by injecting trainable low-rank adapters while freezing most base parameters.
- **Distributed Training**: Robust multi-GPU and multi-node support, handling communication setup and synchronization to simplify scaling.
- **Configuration and Reproducibility**: Structured YAML configs capture datasets, base HF model, optimizer, schedulers, precision/quantization, and PEFT settings to ensure runs are versionable and reproducible.

NeMo Automodel is part of the broader NeMo framework, a comprehensive platform for building custom generative AI models. Built on NVIDIA GPU-accelerated technologies, NeMo scales from a single GPU to large multi-node clusters. Automodel acts as a bridge that lets you use Hugging Face models while benefiting from NeMo's performance and scaling optimizations.



## How it Works

1. **Instantiate a Hugging Face model** using the Automodel interface.
2. **Prepare the dataset** with Hugging Face `datasets` and NeMo utilities.
3. **Configure training** in YAML, including model, data, parallelism (e.g., DDP, FSDP2), and fine-tuning method (e.g., LoRA).
4. **Run the training job**, leveraging NeMo's optimizations and distributed processing.
5. **Deploy** the fine-tuned model, optionally exporting to inference libraries such as vLLM or TensorRT-LLM for high-performance serving.

## Why Day-0 access matters

As organizations strive to maximize the value of their generative AI investments, accessing the latest model developments is crucial to continued success. By using state-of-the-art models on Day-0, teams can harness these innovations efficiently, maintain relevance, and remain competitive.

The past year has seen a flurry of open-source model releases, including Meta Llama, Google Gemma, Mistral Codestral, Codestral Mamba, Large 2, Mixtral, Qwen 3/2/2.5, DeepSeek R1, NVIDIA Nemotron, and NVIDIA Llama Nemotron. These models are often made available on the Hugging Face Hub, providing the broader community with easy access.

Shortly after release, many users evaluate model capabilities and explore potential applications. Fine-tuning for specific use cases quickly becomes a priority to understand model potential and identify opportunities for innovation.

NeMo uses NVIDIA Megatron-Core and Transformer Engine (TE) backends to achieve high throughput and Model FLOPs Utilization (MFU) across thousands of NVIDIA GPUs. Integrating a brand-new architecture directly into the Megatron-Core path may require multi-stage conversions and validation across SFT/PEFT, evaluation, and HF⇄NeMo conversions, introducing a time delay between a model’s release and an optimal recipe. Automodel closes this gap by providing Day-0 support for HF models while preserving a smooth path to Megatron-Core when available.

## Scope and roadmap

Automodel currently supports text generation (LLMs) and vision-language models (VLMs), with plans to extend into additional categories such as video generation.

## Capabilities at a glance

- FSDP2, DDP; TP/CP roadmap
- JIT paths; optimized attention
- Easy opt-in to Megatron-Core when available
- Export to vLLM; TensorRT-LLM planned
- Native HF support; popular models gain Megatron-Core recipes

## Backends: Megatron-Core vs Automodel

For an in-depth comparison, see Key Features. In short: Megatron-Core offers peak throughput and full 4D parallelism; Automodel offers Day‑0 breadth with strong performance and simpler setup.

## Quick start

For a hands-on walkthrough, see the {ref}`get-started-quick-start` guide to fine-tune HF models with Automodel and optionally switch to the Megatron‑Core backend.

## Extending Automodel

For guidance on adding new task-specific Automodel classes, see the Quick Start section on extending Automodel.

## Conclusion

Automodel enables rapid experimentation with performant, native HF model support—no conversions required—and provides a seamless “opt-in” path to the high-performance Megatron-Core stack with minimal code changes. Automodel was introduced with the NeMo 25.02 release. Refer to the tutorial notebooks for PEFT LoRA, SFT, and multinode scaling, and consider contributing feedback and extensions.