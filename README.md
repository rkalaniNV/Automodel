<div align="center">

# 🚀 NeMo AutoModel

</div>

<div align="center">

<!-- [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) -->
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![GitHub Stars](https://img.shields.io/github/stars/NVIDIA/NeMo-Automodel.svg?style=social&label=Star&maxAge=2592000)](https://github.com/NVIDIA-NeMo/Automodel/stargazers/)

**Zero-day integration with Hugging Face models automating fine-tuning and pretraining with built-in parallelism, custom-kernels and optimized recipes**

[📖 Documentation](https://github.com/NVIDIA-NeMo/Automodel/) • [🔥 Quickstart](#quickstart) • [💡 Examples](#examples) • [🤝 Contributing](CONTRIBUTING.md)

</div>

---

## ✨ What is NeMo AutoModel?

NeMo AutoModel delivers true **zero-day**, drop-in compatibility for Hugging Face models. Start fine-tuning models instantly, scale effortlessly with built-in data/model parallelism, optimized custom kernels, and memory-efficient recipes-all while preserving the original checkpoint format for seamless use across the Hugging Face ecosystem.






### 📋 Ready-to-Use Recipes

| Domain | Domain | Model-id | Single-GPU | Single-Node | Multi-Node |
|--------|--------|----------|------------|-------------|------------|
| **LLM** | Llama-3.1 | [`8B-Instruct`](https://huggingface.co/eta-llama/Llama-3.1-8B-Instruct) | [HellaSwag + LoRA](recipes/llm/llama_3_2_1b_hellaswag_peft.yaml) |[HellaSwag](recipes/llm/llama_3_2_1b_hellaswag.yaml) | Coming Soon |
| **LLM** | Llama-3.2 | [`1B`](https://huggingface.co/meta-llama/Llama-3.2-1B), [`1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), [`3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | [HellaSwag + LoRA](recipes/llm/llama_3_2_1b_hellaswag_peft.yaml) |[HellaSwag](recipes/llm/llama_3_2_1b_hellaswag.yaml) | Coming Soon |
| **LLM** | Mistral | [`7B-Instruct-v0.2`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2),[`7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | [HellaSwag + LoRA](recipes/llm/llama_3_2_1b_hellaswag_peft.yaml) |[HellaSwag](recipes/llm/llama_3_2_1b_hellaswag.yaml) | Coming Soon |

| **LLM** | Qwen3 | [`Qwen/Qwen3-32B`](https://huggingface.co/Qwen/Qwen3-32B), [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) | [HellaSwag + LoRA](recipes/llm/llama_3_2_1b_hellaswag_peft.yaml) |[HellaSwag](recipes/llm/llama_3_2_1b_hellaswag.yaml) | Coming Soon |
| **LLM** | Qwen2.5 | [`14B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct), [`7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), [`3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), [`1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | [HellaSwag + LoRA](recipes/llm/llama_3_2_1b_hellaswag_peft.yaml) |[HellaSwag](recipes/llm/llama_3_2_1b_hellaswag.yaml) • [SQuAD](recipes/
| **LLM** | Qwen2 | [`7B-Instruct`](https://huggingface.co/Qwen/Qwen2-7B-Instruct) | [HellaSwag + LoRA](recipes/llm/llama_3_2_1b_hellaswag_peft.yaml) |[HellaSwag](recipes/llm/llama_3_2_1b_hellaswag.yaml) | Coming Soon |



| **LLM** | Phi4 | [`phi-4`](https://huggingface.co/microsoft/phi-4) | [HellaSwag + LoRA](recipes/llm/llama_3_2_1b_hellaswag_peft.yaml) |[HellaSwag](recipes/llm/llama_3_2_1b_hellaswag.yaml) | Coming Soon |
| **LLM** | Phi3 | [`Phi-3-mini-4k-instruct`](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) | [HellaSwag + LoRA](recipes/llm/llama_3_2_1b_hellaswag_peft.yaml) |[HellaSwag](recipes/llm/llama_3_2_1b_hellaswag.yaml) | Coming Soon |
| **LLM** | Phi2 | [`phi-2`](https://huggingface.co/microsoft/phi-2) | [HellaSwag + LoRA](recipes/llm/llama_3_2_1b_hellaswag_peft.yaml) |[HellaSwag](recipes/llm/llama_3_2_1b_hellaswag.yaml) | Coming Soon |


| **VLM** | Gemma3    | [`4b-it`](https://huggingface.co/google/gemma-3-4b-it), [`27b-it`](https://huggingface.co/google/gemma-3-27b-it) | [CORD-v2 + LoRA](recipes/vlm/gemma_3_vl_3b_cord_v2_peft.yaml) | [CORD-v2](recipes/vlm/gemma_3_vl_3b_cord_v2.yaml) | Coming Soon |
| **VLM** | Qwen2.5-VL| [`3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), [`7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | [CORD-v2 + LoRA](recipes/vlm/gemma_3_vl_3b_cord_v2_peft.yaml) | [CORD-v2](recipes/vlm/gemma_3_vl_3b_cord_v2.yaml) | Coming Soon |
| **VLM** | Llama-3.2 | [`11B-Vision`](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision) | [CORD-v2 + LoRA](recipes/vlm/gemma_3_vl_3b_cord_v2_peft.yaml) | [CORD-v2](recipes/vlm/gemma_3_vl_3b_cord_v2.yaml) | Coming Soon |
| **VLM** | Phi-3.5   | [`vision-instruct`](https://huggingface.co/microsoft/Phi-3.5-vision-instruct) | [CORD-v2 + LoRA](recipes/vlm/gemma_3_vl_3b_cord_v2_peft.yaml) | [CORD-v2](recipes/vlm/gemma_3_vl_3b_cord_v2.yaml) | Coming Soon |


### 🚀 Key Features

- **🔥 Day-0 Hugging Face Support**: Any model on Hugging Face Hub works instantly with zero configuration
- **⚡ Lightning Fast Performance**: Custom CUDA kernels deliver 2-5x speedups
- **🌐 Large-Scale Distributed Training**: Built-in FSDP2 and nvFSDP for seamless multi-node scaling
- **👁️ Vision-Language Model Ready**: Native support for VLMs (Qwen2-VL, Gemma-3-VL, etc)
- **🧩 Advanced PEFT Methods**: LoRA and extensible PEFT system out of the box
- **📦 Seamless HF Ecosystem**: Fine-tuned models work perfectly with Transformers pipeline, VLM, etc.
- **⚙️ Production-Grade Infrastructure**: Checkpointing, logging, monitoring, and resiliency
- **🎯 Optimized Recipes**: Pre-built configurations for common models and datasets
- **🔧 Flexible Configuration**: YAML-based configuration system for reproducible experiments
- **⚡ FP8 Precision**: Native FP8 training & inference for higher throughput and lower memory use
- **🔢 INT4 / INT8 Quantization**: Turn-key quantization workflows for ultra-compact, low-memory training


---

## 🔧 Installation

### 🐳 Recommended: NeMo Framework Container (Easiest)

The fastest way to get started is using the pre-built NeMo Framework Training container with NeMo AutoModel already installed:

```bash
# Pull the latest NeMo Framework container
docker pull nvcr.io/nvidia/nemo:25.07

# Run with GPU support
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    nvcr.io/nvidia/nemo:25.07 bash

# Verify installation
python -c "import nemo_automodel; print('✅ NeMo AutoModel ready!')"
```

### 🔧 Install from Source (Development)

```bash
# Clone the repository
git clone https://github.com/NVIDIA-NeMo/Automodel.git
cd NeMo-Automodel

# Install in development mode
pip install -e .

```

---

## 🔥 Quickstart

<!-- ### 30-Second Fine-tuning

```python
import nemo_automodel as na

# Load any Hugging Face model
model = na.NeMoAutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Apply LoRA with one line
na.peft.lora(model, rank=16, alpha=32)

# Your model is ready for training!
``` -->

### Run with Pre-built Recipes

```bash
# Fine-tune LLaMA on HellaSwag (single GPU)
python recipes/llm/finetune.py --config recipes/llm/llama_3_2_1b_squad.yaml

# Fine-tune with LoRA (memory efficient)
python recipes/llm/finetune.py --config recipes/llm/llama_3_2_1b_hellaswag_peft.yaml

# Multi-GPU with FSDP2
torchrun --nproc-per-node=8 recipes/llm/finetune.py --config recipes/llm/llama_3_2_1b_hellaswag.yaml

#Multi-GPU with nvFSDP
torchrun --nproc-per-node=8 recipes/llm/finetune.py --config recipes/llm/llama_3_2_1b_hellaswag_nvfsdp.yaml

# #Multi-Node training
# torchrun --nproc-per-node=8 --nnodes=2 \
#     recipes/llm/finetune.py --config recipes/llm/llama_3_2_1b_squad_nvfsdp.yaml
```

### Vision-Language Models

```bash
# Fine-tune Qwen2.5-VL
python recipes/vlm/finetune.py --config recipes/vlm/qwen2_5_vl_3b_rdr.yaml

# Fine-tune Gemma-3-VL with LoRA
python recipes/vlm/finetune.py --config recipes/vlm/gemma_3_vl_3b_cord_v2_peft.yaml
```

---

## 📋 Examples


### 1. Distributed Training Configuration

```yaml
distributed:
  _target_: nemo_automodel.distributed.nvfsdp.NVFSDPManager
  dp_size: 8
  tp_size: 1
  cp_size: 1

```

### 2. LoRA Configuration
```yaml
peft:
  peft_fn: nemo_automodel._peft.lora.apply_lora_to_linear_modules
  match_all_linear: True
  dim: 8
  alpha: 32
  use_triton: True
```

### 3. Vision-Language Model Fine-tuning
```yaml
model:
  _target_: nemo_automodel._transformers.NeMoAutoModelForImageTextToText.from_pretrained
  pretrained_model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct
  
processor:
  _target_: transformers.AutoProcessor.from_pretrained
  pretrained_model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct
  min_pixels: 200704
  max_pixels: 1003520 
```

### 4. Checkpointing & Resume
```yaml
checkpoint:
  enabled: true
  checkpoint_dir: ./checkpoints
  save_consolidated: true      # HF-compatible safetensors
  model_save_format: safetensors
```

---

## 🎛️ Supported Models

### Large Language Models
- **LLaMA Family**: LLaMA 3, LLaMA 3.1, LLaMA 3.2, Code Llama
- **And more**: Any causal LM on Hugging Face Hub!

### Vision-Language Models  
- **Qwen2.5-VL**: All variants (3B, 7B, 72B)
- **Gemma-3-VL**: 3B and other variants

### PEFT Methods
- **LoRA**: Low-Rank Adaptation
<!-- - **DoRA**: Weight-Decomposed Low-Rank Adaptation   -->
- **Custom**: Easy to implement new PEFT methods

---

<!-- ## ⚡ Performance (Do we have a table like to show/do we want to show it?)

NeMo AutoModel delivers significant speedups through optimized kernels and distributed training:

| Model | Method | Speedup | Memory Savings |
|-------|--------|---------|----------------|
| LLaMA-3-8B | LoRA + Liger | **3.2x** | 60% |
| Qwen2.5-7B | Full FT + FSDP2 | **2.8x** | 40% |
| Gemma-2-9B | DoRA + Cut-CE | **4.1x** | 55% |

### Optimizations Included
- **Liger Kernel**: Optimized attention and MLP operations
- **Cut-CrossEntropy**: Memory-efficient loss computation
- **FSDP2**: Latest fully sharded data parallelism
- **nvFSDP**: NVIDIA's enterprise FSDP implementation
- **Mixed Precision**: Automatic FP16/BF16 training

--- -->

## 🗂️ Project Structure

```
NeMo-Automodel/
├── nemo_automodel/              # Core library
│   ├── _peft/                   # PEFT implementations (LoRA)
│   ├── _transformers/           # HF model integrations
│   ├── checkpoint/              # Distributed checkpointing
│   ├── datasets/                # Dataset loaders
│   │   ├── llm/                 # LLM datasets (HellaSwag, SQuAD, etc.)
│   │   └── vlm/                 # VLM datasets (CORD-v2, rdr etc.)
│   ├── distributed/             # FSDP2, nvFSDP, parallelization
│   ├── loss/                    # Optimized loss functions
│   └── training/                # Training recipes and utilities
├── recipes/                     # Ready-to-use training recipes
│   ├── llm/                     # LLM fine-tuning recipes
│   └── vlm/                     # VLM fine-tuning recipes
└── tests/                       # Comprehensive test suite
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

---

## 📄 License

<!-- This project is licensed under the ??? - see the [LICENSE](LICENSE) file for details. -->

---

## 🔗 Links

- **Documentation**: https://docs.nvidia.com/nemo-framework/user-guide/latest/automodel/index.html
- **NVIDIA NeMo Framework**: https://github.com/NVIDIA/NeMo
- **Hugging Face Hub**: https://huggingface.co/models
- **Issues**: https://github.com/NVIDIA-NeMo/Automodel/issues
- **Discussions**: https://github.com/NVIDIA-NeMo/Automodel/discussions

---

<div align="center">

**Made with ❤️ by NVIDIA**

*Accelerating AI for everyone*

</div>
