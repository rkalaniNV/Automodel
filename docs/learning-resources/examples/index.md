# Examples

Task-focused examples demonstrating NeMo AutoModel's competitive advantages for AI development teams. Each example solves real production challenges with verified performance improvements.

## Overview

These examples show how to accomplish common AI training tasks with NeMo AutoModel's optimizations. Every example includes working configurations, performance benchmarks, and practical deployment guidance based on actual codebase capabilities.

## Training Examples by Task

:::::{grid} 1 1 2 2
:gutter: 2

::::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` High-Performance Text Classification
:link: high-performance-text-classification
:link-type: doc
:link-alt: High-performance text classification

Get 2-3x PyTorch speedup with automatic optimizations. Perfect drop-in replacement for HF Trainer workflows.
+++
{bdg-primary}`LLM`
::::

::::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Memory-Efficient Large Model Training
:link: memory-efficient-training
:link-type: doc
:link-alt: Memory-efficient large model training

Train 7B+ models on mainstream GPUs using PEFT and distributed strategies. Breakthrough memory limitations.
+++
{bdg-info}`LLM`
::::

::::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Multi-Node Distributed Training
:link: distributed-training
:link-type: doc
:link-alt: Multi-node distributed training

Production Slurm integration for enterprise-scale training. Built-in monitoring and job management.
+++
{bdg-warning}`Distributed`
::::

::::{grid-item-card} {octicon}`eye;1.5em;sd-mr-1` Advanced Multi-Modal Fine-Tuning
:link: multimodal-finetuning
:link-type: doc
:link-alt: Advanced multi-modal fine-tuning

Vision-language model optimization with custom datasets and experimental techniques for research workflows.
+++
{bdg-secondary}`VLM`
::::

:::::

## Real-World Performance Impact

**Immediate Value Delivered:**
- **Training Speed**: 2-3x faster than vanilla PyTorch with zero code changes
- **Memory Efficiency**: Train 7B models on 8GB GPUs through optimized PEFT
- **Infrastructure Cost**: 60-80% reduction in GPU training time
- **Enterprise Ready**: Production deployment with monitoring and compliance

(high-performance-text-classification)=
## High-Performance Text Classification

**Task**: Replace HF Trainer workflows with automatic performance optimizations

### [Optimized Sentiment Analysis with Performance Benchmarking](high-performance-text-classification.md)

Demonstrate 2-3x speedup over vanilla PyTorch using sentiment analysis as a practical example. Includes real performance benchmarks with Liger kernels, SDPA, and Flash Attention optimizations that actually exist in the codebase.

**Techniques**: Automatic optimization, performance benchmarking, workflow migration  
**Suitable for**: Applied ML Engineers, Infrastructure-Aware Developers  
**Hardware**: Single GPU (8GB+)  
**Key Value**: Immediate workflow acceleration with zero code changes

(memory-efficient-training)=
## Memory-Efficient Large Model Training  

**Task**: Train large models on resource-constrained hardware

### [7B Model Training on Consumer GPUs with PEFT](memory-efficient-training.md)

Train large language models using verified PEFT implementations and distributed strategies. Includes multi-modal scaling examples and memory optimization techniques for production environments.

**Techniques**: PEFT with LoRA, FSDP2/nvFSDP, memory optimization, VLM scaling  
**Suitable for**: Infrastructure-Aware Developers, Enterprise Practitioners  
**Hardware**: Single/Multi-GPU (8GB+ per GPU)  
**Key Value**: 2-3x larger models on same hardware

(distributed-training)=
## Multi-Node Distributed Training

**Task**: Deploy enterprise-scale training across cluster infrastructure  

### [Production Slurm Integration with Enterprise Monitoring](distributed-training.md)

Production-ready multi-node training using actual Slurm integration capabilities. Includes enterprise monitoring, job management, and containerization features that exist in the codebase.

**Techniques**: Slurm integration, multi-node scaling, enterprise monitoring, containerization  
**Suitable for**: Enterprise Practitioners, Infrastructure-Aware Developers  
**Hardware**: Multi-node cluster with Slurm  
**Key Value**: Production-grade ML infrastructure

(multimodal-finetuning)=
## Advanced Multi-Modal Fine-Tuning

**Task**: Push vision-language models beyond standard configurations

### [Custom VLM Training with Experimental Optimizations](multimodal-finetuning.md)

Advanced vision-language model fine-tuning with custom dataset integration and cutting-edge optimization techniques. Explores research capabilities beyond standard frameworks.

**Techniques**: Advanced VLM training, custom datasets, experimental optimizations, freeze strategies  
**Suitable for**: Open-Source Enthusiasts, Applied ML Engineers  
**Hardware**: Multi-GPU setup recommended  
**Key Value**: Research capabilities beyond standard frameworks

## Ready-to-Run Code Examples

**Looking for working configurations you can run immediately?** The repository includes a comprehensive collection of production-ready examples with complete YAML configurations and execution scripts.

:::::{grid} 1 1 2 2
:gutter: 2

::::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` LLM Fine-Tuning Examples
:link: https://github.com/NVIDIA/NeMo-Automodel/tree/main/examples/llm
:link-alt: LLM examples directory

**8 complete configurations** covering Llama 3.2, Qwen models with PEFT, nvFSDP, FP8 optimizations
+++
{bdg-primary}`Ready-to-run` {bdg-secondary}`examples/llm/`
::::

::::{grid-item-card} {octicon}`device-camera-video;1.5em;sd-mr-1` Vision-Language Model Examples  
:link: https://github.com/NVIDIA/NeMo-Automodel/tree/main/examples/vlm
:link-alt: VLM examples directory

**10 configurations** for Gemma, Phi, Qwen2.5 VL models with fine-tuning and generation scripts
+++
{bdg-success}`Production-ready` {bdg-secondary}`examples/vlm/`
::::

:::::

### Quick Start with Code Examples

1. **Navigate to** `examples/llm/` or `examples/vlm/` in your installation
2. **Choose a YAML configuration** that matches your model and optimization needs  
3. **Run using** the provided `finetune.py` scripts
4. **Customize** the YAML settings for your specific dataset and requirements

**Recommended Starting Points:**
- **LLM SFT**: `examples/llm/llama_3_2_1b_squad.yaml`
- **PEFT Training**: `examples/llm/llama_3_2_1b_hellaswag_peft.yaml` 
- **VLM Fine-tuning**: `examples/vlm/gemma_3_vl_4b_medpix_peft.yaml`

## How to Use Tutorial Examples

1. **Navigate to the examples directory** in your NeMo Automodel installation
2. **Choose an example** that matches your model type and use case
3. **Review the YAML configuration** to understand the settings
4. **Run the example** using the provided Python scripts
5. **Modify settings** to adapt the example to your specific needs

## Example Structure

Each example typically includes:
- **YAML Configuration**: Complete training configuration
- **Python Script**: Execution script with proper imports
- **Documentation**: Explanation of settings and expected outcomes
- **Requirements**: Specific dependencies and hardware requirements

## Get Started with Examples

For first-time users, we recommend starting with:

1. **LLM SFT Example**: `examples/llm/llama_3_2_1b_squad.yaml`
2. **PEFT Example**: `examples/llm/llama_3_2_1b_hellaswag_peft.yaml`
3. **VLM Example**: `examples/vlm/gemma_3_vl_4b_medpix_peft.yaml`

These examples provide a solid foundation for understanding NeMo Automodel's capabilities and can be easily adapted for your specific requirements.

## Choose the Right Example

**For Performance Optimization** → High-Performance Text Classification  
**For Memory Constraints** → Memory-Efficient Large Model Training  
**For Enterprise Deployment** → Multi-Node Distributed Training  
**For Research & Experimentation** → Advanced Multi-Modal Fine-Tuning

## Get Started

1. **Choose an example** based on your primary goal and constraints
2. **Review hardware requirements** and ensure compatibility  
3. **Follow the step-by-step guide** with working configurations
4. **Benchmark performance** against your current workflows
5. **Adapt configurations** for your specific use cases

```{toctree}
:maxdepth: 2
:hidden:

high-performance-text-classification
memory-efficient-training
distributed-training
multimodal-finetuning
```
