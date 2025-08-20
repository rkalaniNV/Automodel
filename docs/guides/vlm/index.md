---
description: "Learn how to fine-tune vision language models using NeMo Automodel for multi-modal tasks that combine visual and textual understanding."
categories: ["model-training"]
tags: ["vlm", "multimodal", "fine-tuning", "vision", "yaml-config", "recipes"]
personas: ["mle-focused", "researcher-focused", "data-scientist-focused"]
difficulty: "intermediate"
content_type: "concept"
modality: "vlm"
---

(vlm-training-overview)=
# Train Vision Language Models

Learn how to fine-tune Vision Language Models (VLMs) using NeMo Automodel for multi-modal tasks that combine visual and textual understanding.

(vlm-overview)=
## Overview

Vision Language Models enable AI systems to understand and reason about both visual and textual information simultaneously. NeMo Automodel provides comprehensive support for training VLMs with multi-modal datasets, making it easy to build models that can:

- Answer questions about images (Visual Question Answering)
- Generate captions for images
- Understand multi-modal conversations
- Process medical, scientific, or domain-specific visual content

(vlm-multimodal-training)=
## Multi-Modal Training

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`image;1.5em;sd-mr-1` Dataset Integration
:link: dataset
:link-type: doc
:link-alt: Multi-modal dataset guide

Learn how to integrate your own multi-modal datasets combining text with images, audio, or other modalities for VLM training.
+++
{bdg-primary}`Multi-modal`
{bdg-secondary}`Custom Data`
:::

:::{grid-item-card} {octicon}`question;1.5em;sd-mr-1` Visual QA Training
:link: dataset
:link-type: doc
:link-alt: Visual question answering

Train models for visual question answering tasks with specialized preprocessing and formatting for VQA datasets.
+++
{bdg-info}`VQA`
{bdg-secondary}`Question Answering`
:::

::::

(vlm-supported-modalities)=
## Supported Modalities

- **Images**: JPEG, PNG, and other standard image formats
- **Audio**: Speech and audio processing capabilities  
- **Text**: Natural language instructions, questions, and responses
- **Multi-turn Conversations**: Complex dialogue scenarios with visual context

(vlm-key-features)=
## Key Features

- **Flexible Data Processing**: Support for various multi-modal dataset formats
- **Custom Collation Functions**: Specialized batching for diverse data types
- **Chat Template Integration**: Proper formatting for multi-turn dialogues
- **PEFT Support**: Efficient fine-tuning with LoRA for large VLMs
- **Distributed Training**: Scale VLM training across multiple GPUs with FSDP2/nvFSDP

(vlm-use-cases)=
## Example Use Cases

- **Medical VQA**: Train models on medical imaging datasets like MedPix-VQA
- **Educational Content**: Visual question answering for educational materials
- **Accessibility**: Image description and analysis for visual accessibility
- **Scientific Research**: Analysis of scientific figures and diagrams

```{toctree}
:maxdepth: 2
:hidden:

dataset
```
