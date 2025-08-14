(about-key-features)=
# Key Features

## Core Features and Capabilities

### "Day-0" Hugging Face Integration
Automodel provides immediate support for newly released and existing models on the Hugging Face Hub without requiring conversions or checkpoint rewrites, saving significant development time. It integrates seamlessly with the Hugging Face ecosystem, including `datasets`, tokenizers, and related tooling.

### Accelerated Performance
Engineered for speed and efficiency using optimized attention paths, fused kernels, and memory‑saving strategies, Automodel delivers substantial speedups compared to stock training loops. It supports mixed precision (e.g., BF16) and FP8 quantization on supported hardware to reduce memory usage and increase throughput.

### Large-Scale Distributed Training
Automodel includes native distributed support for training across multiple GPUs and nodes. It integrates with PyTorch‑native parallelisms like Distributed Data Parallel (DDP), NeMo's FSDP2, and nvFSDP to efficiently scale to billion‑parameter models.

### Fine-Tuning and Optimization
Automodel simplifies customization via both Supervised Fine-Tuning (SFT) and Parameter-Efficient Fine-Tuning (PEFT) techniques such as LoRA, enabling strong task adaptation with a minimal number of trainable parameters.

### Ready-to-Use Recipes and Configuration
Automodel offers ready-to-use recipes that define end-to-end workflows (data preparation, training, evaluation). These are easily customized through a flexible YAML-based configuration system to match single-GPU or multi-GPU setups.

## Capabilities at a glance

- Model and data parallelism: FSDP2 and DDP today; TP and CP on the roadmap
- Enhanced PyTorch performance with JIT compilation paths
- Seamless transition to Megatron-Core optimized training/post-training recipes as they become available
- Export to vLLM for optimized inference; TensorRT-LLM export is planned
- Native HF integration without checkpoint rewrites; popular models gain optimized Megatron-Core support over time

## Backends: Megatron-Core vs Automodel

| Aspect | Megatron-Core Backend | Automodel Backend |
| --- | --- | --- |
| Coverage | Most popular LLMs with expert-tuned recipes | All HF text models on Day-0 |
| Training throughput | Optimal throughput with Megatron-Core kernels | Good performance with Liger kernels, cut cross-entropy, and PyTorch JIT |
| Scalability | Up to 1,000+ GPUs with full 4D parallelism (TP, PP, CP, EP) | Comparable scale using PyTorch-native TP, CP, and FSDP2 at slightly reduced throughput |
| Inference path | Export to TensorRT-LLM, vLLM, or NVIDIA NIM | Export to vLLM |

