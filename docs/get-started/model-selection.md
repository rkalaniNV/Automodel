---
description: "Choose the right model for your use case from NeMo Automodel's extensive catalog of supported language models, vision language models, and speech models."
tags: ["model-selection", "llm", "vlm", "asr", "hugging-face", "planning"]
categories: ["getting-started"]
personas: ["mle-focused", "data-scientist-focused", "researcher-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "universal"
---

(get-started-model-selection)=
# Choose Your Model

Choose the right model for your task from NeMo Automodel's extensive catalog of supported architectures. This guide helps you navigate the options and make informed decisions based on your use case, hardware, and performance requirements.

## Overview

NeMo Automodel seamlessly integrates with Hugging Face Hub, providing Day-0 support for new models with optimized training workflows. You can fine-tune models across three main categories:

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} {octicon}`comment-discussion;1.5em;sd-mr-1` Language Models
:class-header: sd-bg-primary sd-text-white

**Text generation and understanding**
- Code generation
- Instruction following  
- Conversational AI
- Text summarization
:::

:::{grid-item-card} {octicon}`image;1.5em;sd-mr-1` Vision Language Models
:class-header: sd-bg-info sd-text-white

**Multimodal understanding**
- Image captioning
- Visual question answering
- Document analysis
- Multimodal reasoning
:::

:::{grid-item-card} {octicon}`unmute;1.5em;sd-mr-1` Speech Models
:class-header: sd-bg-success sd-text-white

**Audio processing**
- Speech transcription
- Audio understanding
- Speech-to-text
- Multilingual ASR
:::

::::

## Large Language Models (LLMs)

### LLaMA Family

Powerful general-purpose language models from Meta, excellent for most text tasks.

```{list-table}
:header-rows: 1
:widths: 25 15 25 35

* - Model
  - Size
  - Use Case
  - Example HF Model
* - **LLaMA 3.2**
  - 1B, 3B
  - Lightweight chat, edge deployment
  - `meta-llama/Llama-3.2-1B`
* - **LLaMA 3.1**
  - 8B, 70B
  - General chat, instruction following
  - `meta-llama/Llama-3.1-8B-Instruct`
* - **Code Llama**
  - 7B, 13B, 34B
  - Code generation, programming assistance
  - `codellama/CodeLlama-7b-Instruct-hf`
```

**Best for**: General-purpose applications, instruction following, conversational AI

**Example configuration**:
```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B
  torch_dtype: torch.bfloat16
```

### Qwen Family

Alibaba's multilingual models with strong reasoning capabilities.

```{list-table}
:header-rows: 1
:widths: 25 15 25 35

* - Model
  - Size
  - Use Case
  - Example HF Model
* - **Qwen3**
  - 0.5B, 1.5B, 7B
  - Multilingual tasks, efficiency
  - `Qwen/Qwen2.5-0.5B-Instruct`
* - **Qwen2.5**
  - 0.5B-72B
  - Advanced reasoning, math
  - `Qwen/Qwen2.5-7B-Instruct`
* - **Qwen2**
  - 0.5B-72B
  - General purpose, code
  - `Qwen/Qwen2-7B-Instruct`
```

**Best for**: Multilingual applications, mathematical reasoning, code generation

**Example configuration**:
```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: Qwen/Qwen2.5-7B-Instruct
  torch_dtype: torch.bfloat16
```

### Gemma Family

Google's efficient models optimized for safety and instruction following.

```{list-table}
:header-rows: 1
:widths: 25 15 25 35

* - Model
  - Size
  - Use Case
  - Example HF Model
* - **Gemma 3**
  - 2B, 9B
  - Efficient chat, safety
  - `google/gemma-2-2b-it`
* - **Gemma 2**
  - 2B, 9B, 27B
  - Instruction following
  - `google/gemma-2-9b-it`
```

**Best for**: Safety-critical applications, instruction following, resource-constrained environments

### Phi Family

Microsoft's small but capable models for edge deployment.

```{list-table}
:header-rows: 1
:widths: 25 15 25 35

* - Model
  - Size
  - Use Case
  - Example HF Model
* - **Phi 4**
  - 14B
  - Advanced reasoning, STEM
  - `microsoft/Phi-4-Instruct`
* - **Phi 3**
  - 3.8B, 7B, 14B
  - Edge deployment, efficiency
  - `microsoft/Phi-3-mini-4k-instruct`
* - **Phi 2**
  - 2.7B
  - Lightweight applications
  - `microsoft/phi-2`
```

**Best for**: Edge deployment, resource efficiency, mathematical reasoning

## Vision Language Models (VLMs)

### Qwen2.5-VL Family

Advanced multimodal models for vision language understanding.

```{list-table}
:header-rows: 1
:widths: 25 15 25 35

* - Model
  - Size
  - Use Case
  - Example HF Model
* - **Qwen2.5-VL**
  - 3B, 7B, 72B
  - Image understanding, VQA
  - `Qwen/Qwen2.5-VL-3B-Instruct`
* - **Qwen2-VL**
  - 2B, 7B, 72B
  - Document analysis, OCR
  - `Qwen/Qwen2-VL-7B-Instruct`
```

**Best for**: Visual question answering, document understanding, image captioning

**Example configuration**:
```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained
  pretrained_model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct
  torch_dtype: torch.bfloat16
```

### Gemma-3-VL Family

Google's vision language models with safety features.

```{list-table}
:header-rows: 1
:widths: 25 15 25 35

* - Model
  - Size
  - Use Case
  - Example HF Model
* - **Gemma-3-VL**
  - 3B, 4B
  - Safe image understanding
  - `google/gemma-3-4b-it`
```

**Best for**: Safe multimodal applications, content moderation, educational tools

## Speech Models (ASR)

Models for automatic speech recognition and audio understanding.

```{list-table}
:header-rows: 1
:widths: 30 20 50

* - Model Type
  - Example Models
  - Use Case
* - **Whisper Family**
  - `openai/whisper-large-v3`
  - General speech recognition
* - **Wav2Vec2**
  - `facebook/wav2vec2-large-960h`
  - English speech recognition
* - **Multilingual ASR**
  - `openai/whisper-large-v3-turbo`
  - Multiple language support
```

**Best for**: Speech transcription, multilingual audio processing, real-time ASR

## Hardware Considerations

### GPU Memory Requirements

Choose models based on your available GPU memory:

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`cpu;1.5em;sd-mr-1` 8-16GB GPU
:class-header: sd-bg-success sd-text-white

**Recommended Models:**
- LLaMA 3.2 1B-3B
- Qwen 0.5B-1.5B  
- Phi 2-3 mini
- Gemma 2B

**Training:** Full fine-tuning or PEFT
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` 24-48GB GPU
:class-header: sd-bg-info sd-text-white

**Recommended Models:**
- LLaMA 3.1 8B
- Qwen 7B
- Gemma 9B
- VLMs up to 7B

**Training:** Full fine-tuning with FSDP2
:::

:::{grid-item-card} {octicon}`rack-server;1.5em;sd-mr-1` 80GB+ GPU
:class-header: sd-bg-warning sd-text-white

**Recommended Models:**
- LLaMA 3.1 70B
- Qwen 72B
- Large VLMs
- Any model with PEFT

**Training:** Multi-GPU distributed
:::

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` Multi-Node Cluster
:class-header: sd-bg-primary sd-text-white

**Recommended Models:**
- Any size model
- Largest VLMs (72B+)
- Custom architectures

**Training:** nvFSDP, tensor parallelism
:::

::::

### Memory Optimization Strategies

**For limited GPU memory:**

1. **Use PEFT/LoRA**: Reduce trainable parameters by 90%+
2. **Enable gradient checkpointing**: Trade compute for memory
3. **Use FSDP2**: Shard parameters across GPUs
4. **Mixed precision**: Use BF16 or FP16 training

```yaml
# PEFT configuration for memory efficiency
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  dim: 8
  alpha: 32
```

## Task-Based Recommendations

### Text Generation and Chat

**Best choices**: LLaMA 3.1/3.2, Qwen 2.5, Gemma 2

```bash
# Conversational AI
automodel finetune llm -c examples/llm/llama_3_2_1b_squad.yaml

# Code generation  
automodel finetune llm -c examples/llm/qwen_3_0p6b_hellaswag.yaml
```

### Image Understanding

**Best choices**: Qwen2.5-VL, Gemma-3-VL

```bash
# Medical image analysis
automodel finetune vlm -c examples/vlm/gemma_3_vl_4b_medpix.yaml

# General VQA
automodel finetune vlm -c examples/vlm/qwen2_5_vl_3b_rdr.yaml
```

### Document Processing

**Best choices**: Qwen2-VL (excellent OCR), Phi 4 (reasoning)

### Code Generation

**Best choices**: Code Llama, Qwen2.5, Phi 4

### Multilingual Tasks

**Best choices**: Qwen family, LLaMA 3.1 (70B), Gemma 2

## Getting Started Examples

### Small Model (Good for Experimentation)

```bash
# Quick start with 1B model
automodel finetune llm -c examples/llm/llama_3_2_1b_squad.yaml
```

### Production-Ready Model

```bash
# 7B model with PEFT for efficiency
automodel finetune llm -c examples/llm/llama_3_2_1b_hellaswag_peft.yaml
```

### Vision Language Task

```bash
# Multimodal fine-tuning
automodel finetune vlm -c examples/vlm/gemma_3_vl_4b_medpix.yaml
```

## Custom Model Selection

### Using Any Hugging Face Model

You can use any compatible Hugging Face model by modifying the configuration:

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: YOUR_MODEL_NAME_HERE
  torch_dtype: torch.bfloat16
```

**Compatible architectures include**: Most causal language models on Hugging Face Hub

### Model Architecture Support

For the complete list of supported architectures, see {doc}`../model-coverage/index`.

## Decision Framework

### 1. Define Your Task
- **Text-only**: Use LLM
- **Text + Images**: Use VLM  
- **Speech**: Use ASR model

### 2. Consider Hardware
- **Single GPU (8-16GB)**: Small models (1B-3B)
- **Single GPU (24GB+)**: Medium models (7B-9B)
- **Multi-GPU**: Large models (70B+)

### 3. Evaluate Requirements
- **Latency**: Smaller models (Phi, small Gemma)
- **Quality**: Larger models (LLaMA 70B, Qwen 72B)
- **Multilingual**: Qwen family
- **Safety**: Gemma family

### 4. Choose Training Strategy
- **Fast iteration**: PEFT/LoRA
- **Best quality**: Full fine-tuning
- **Limited resources**: Gradient checkpointing + FSDP2

## Next Steps

Once you've selected your model:

1. **Start with examples**: Use provided configuration files
2. **Experiment with PEFT**: Try LoRA for memory efficiency
3. **Scale gradually**: Start small, then increase model size
4. **Monitor resources**: Use {doc}`../references/troubleshooting-reference` for optimization
5. **Explore advanced features**: Check {doc}`../about/architecture-overview` for optimizations

## Learn More

- {doc}`quick-start` - Run your first training job
- {doc}`../model-coverage/index` - Complete model compatibility matrix
- {doc}`../guides/llm/sft` - LLM fine-tuning deep dive
- {doc}`../guides/vlm/index` - Vision-language model training
- {doc}`../references/yaml-configuration-reference` - Configuration options
