(about-key-features)=
# Key Features

This page provides a comprehensive overview of NeMo Automodel's capabilities, performance optimizations, and comparisons with other training backends to help you understand what makes it powerful for modern AI training workflows.

## Core Features and Capabilities

::::{grid} 1 1 2 3
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Day-0 Support
:class-header: sd-bg-primary sd-text-white

Immediate access to new Hugging Face models without conversions or rewrites.
+++
{bdg-primary}`Ready to Use`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Accelerated Training
:class-header: sd-bg-success sd-text-white

BF16/FP8 precision, optimized attention, and Liger kernels for faster training.
+++
{bdg-success}`Performance`
:::

:::{grid-item-card} {octicon}`organization;1.5em;sd-mr-1` Distributed Scaling
:class-header: sd-bg-info sd-text-white

DDP, FSDP2, and multi-node support for scaling from single GPU to clusters.
+++
{bdg-info}`Scalable`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` PEFT Integration
:class-header: sd-bg-warning sd-text-white

Built-in LoRA for efficient fine-tuning with minimal parameter updates.
+++
{bdg-warning}`Efficient`
:::

:::{grid-item-card} {octicon}`file-code;1.5em;sd-mr-1` YAML Configuration
:class-header: sd-bg-secondary sd-text-white

Structured configs for reproducible training and easy experiment management.
+++
{bdg-secondary}`Reproducible`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Export Ready
:class-header: sd-bg-dark sd-text-white

Deploy to vLLM, TensorRT-LLM, and other inference frameworks.
+++
{bdg-dark}`Deployment`
:::

::::

### Day-0 Hugging Face Integration

Automodel provides immediate support for newly released and existing models on the Hugging Face Hub without requiring conversions or checkpoint rewrites, saving significant development time. 

**Key Benefits:**
- **Immediate Model Access**: Use new models as soon as they're released on HF Hub
- **No Conversion Required**: Direct model loading without format changes
- **Ecosystem Integration**: Seamless compatibility with `datasets`, tokenizers, and HF tooling
- **Preserved APIs**: Maintain familiar Hugging Face interfaces and workflows

The rapid pace of open-source model releases—including Meta Llama, Google Gemma, Mistral Codestral, Mixtral, Qwen series, DeepSeek R1, NVIDIA Nemotron, and NVIDIA Llama Nemotron—creates opportunities for immediate experimentation. Automodel eliminates the traditional delay between model release and optimized training recipes.

### Accelerated Performance

Engineered for speed and efficiency using optimized attention paths, fused kernels, and memory‑saving strategies, Automodel delivers substantial speedups compared to stock training loops.

**Performance Optimizations:**
- **SDPA Integration**: Scaled Dot-Product Attention with backend selection
- **Liger Kernels**: Automatic attention layer patching for speed improvements  
- **Flash Attention 2**: Memory-efficient attention implementation
- **Mixed Precision**: BF16 training with automatic loss scaling
- **FP8 Quantization**: Hardware-accelerated training on H100+ GPUs
- **JIT Compilation**: Enhanced PyTorch performance paths

### Large-Scale Distributed Training

Automodel includes native distributed support for training across multiple GPUs and nodes, integrating with PyTorch‑native parallelisms to efficiently scale to billion‑parameter models.

**Distributed Strategies:**
- **DDP (Data Parallel)**: Simple multi-GPU setup with gradient synchronization
- **FSDP2**: Parameter sharding with optional tensor parallelism and CPU offloading
- **nvFSDP**: NVIDIA-optimized FSDP with advanced overlap strategies
- **Multi-Node Support**: Seamless scaling across cluster environments
- **Memory Optimization**: Gradient checkpointing and parameter offloading

### Fine-Tuning and Optimization

Automodel simplifies customization via both Supervised Fine-Tuning (SFT) and Parameter-Efficient Fine-Tuning (PEFT) techniques, enabling strong task adaptation with minimal computational overhead.

**Fine-Tuning Options:**
- **Supervised Fine-Tuning (SFT)**: Full model parameter updates
- **LoRA (Low-Rank Adaptation)**: Efficient fine-tuning with low-rank matrices
- **DoRA**: Enhanced LoRA variant with improved performance
- **Module Targeting**: Flexible pattern matching for layer selection
- **Freezing Strategies**: Configurable parameter freezing (embeddings, vision towers, etc.)

### Ready-to-Use Recipes and Configuration

Automodel offers ready-to-use recipes that define end-to-end workflows (data preparation, training, evaluation) with flexible YAML-based configuration for reproducible experiments.

**Configuration Features:**
- **YAML-Driven**: Structured configuration with `_target_` pattern instantiation
- **Modular Design**: Mix and match components for custom workflows
- **Example Configs**: Production-ready configurations for popular models
- **Reproducible Runs**: Version-controlled configurations with experiment tracking

## Capabilities at a glance

- Model and data parallelism: FSDP2 and DDP today; TP and CP on the roadmap
- Enhanced PyTorch performance with JIT compilation paths
- Seamless transition to Megatron-Core optimized training/post-training recipes as they become available
- Export to vLLM for optimized inference; TensorRT-LLM export is planned
- Native HF integration without checkpoint rewrites; popular models gain optimized Megatron-Core support over time

## Backend Comparison: Megatron-Core vs Automodel

Understanding when to use Automodel versus Megatron-Core helps you choose the right approach for your training needs. In short: **Megatron-Core offers peak throughput and full 4D parallelism; Automodel offers Day‑0 breadth with strong performance and simpler setup.**

### Detailed Comparison

```{list-table}
:header-rows: 1
:widths: 25 35 40

* - Aspect
  - Megatron-Core Backend
  - Automodel Backend
* - **Model Coverage**
  - Most popular LLMs with expert-tuned recipes
  - All HF text models on Day-0, including newest releases
* - **Training Throughput**
  - Optimal throughput with Megatron-Core kernels
  - Strong performance with Liger kernels, optimized attention, and PyTorch JIT
* - **Scalability**
  - Up to 1,000+ GPUs with full 4D parallelism (TP, PP, CP, EP)
  - Comparable scale using PyTorch-native TP, CP, and FSDP2 at slightly reduced throughput
* - **Setup Complexity**
  - Requires model-specific recipes and configurations
  - Simple YAML configs work across model families
* - **Development Speed**
  - May require waiting for model-specific optimizations
  - Immediate experimentation with any HF model
* - **Memory Efficiency**
  - Highly optimized memory usage patterns
  - Good memory efficiency with FSDP2 and gradient checkpointing
* - **Fine-tuning Support**
  - Full fine-tuning with expert configurations
  - Both SFT and PEFT with flexible targeting
* - **Inference Path**
  - Export to TensorRT-LLM, vLLM, or NVIDIA NIM
  - Export to vLLM; TensorRT-LLM planned
* - **Best Use Cases**
  - Production training at scale with proven models
  - Research, experimentation, and new model evaluation
```

### When to Choose Each Backend

**Choose Megatron-Core when:**
- Training at 1,000+ GPU scale
- Using well-supported model architectures (Llama, Mistral, etc.)
- Maximizing training throughput is critical
- Production workloads with established models

**Choose Automodel when:**
- Experimenting with newly released models
- Need Day-0 support for any HF model
- Rapid prototyping and research workflows  
- Simpler setup and configuration preferences
- PEFT/LoRA fine-tuning workflows

### Migration Path

Automodel provides a smooth transition path to Megatron-Core optimizations as they become available for your model architecture, allowing you to start immediately and optimize later.

