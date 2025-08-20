---
description: "Get up and running with NeMo Automodel in minutes. Learn to fine-tune your first model using CLI and Python approaches."
categories: ["getting-started"]
tags: ["quickstart", "fine-tuning", "automodel-cli", "python-api", "yaml-config", "huggingface"]
personas: ["mle-focused", "researcher-focused", "data-scientist-focused"]
difficulty: "beginner"
content_type: "tutorial"
modality: "universal"
---

(get-started-quick-start)=
# Quick Start

Get up and running with NeMo Automodel in minutes. This guide shows you how to fine-tune your first model using both CLI and Python approaches.

## How It Works

AutoModel follows a simple four-step workflow:

1. **Load a Hugging Face model** using AutoModel's drop-in interface
2. **Prepare your dataset** with Hugging Face datasets and NeMo utilities  
3. **Configure training** via YAML with model, data, parallelism, and fine-tuning settings
4. **Train and deploy** leveraging NeMo's optimizations and export to inference frameworks

(quick-start-prerequisites)=
## Prerequisites

- **Python**: 3.10+
- **GPU**: NVIDIA GPU with 8GB+ memory
- **CUDA**: CUDA Toolkit 11.8+ or 12.x
- **PyTorch**: 2.0+ with CUDA support
- **Storage**: ~5GB for model and checkpoints

## Installation

Install NeMo Automodel from source:

```bash
git clone https://github.com/NVIDIA/NeMo-Automodel.git
cd NeMo-Automodel
pip install -e .
```

For other installation options, refer to the [Installation Guide](../get-started/installation.md).

(cli-approach)=
## Approach 1: CLI (Recommended)

The fastest way to get started is with the `automodel` CLI:

### Single GPU Training

```bash
automodel finetune llm -c examples/llm/llama_3_2_1b_squad.yaml
```

### Multi-GPU Training

```bash
automodel finetune llm -c examples/llm/llama_3_2_1b_squad.yaml --nproc-per-node=2
```

The CLI automatically:

- Downloads the Llama 3.2 1B model from Hugging Face
- Loads the SQuAD dataset for question-answering fine-tuning
- Configures FSDP2 for efficient distributed training
- Saves checkpoints and final model

(python-recipe-approach)=
## Approach 2: Python Recipe

For more control, use the Python recipe directly:

```bash
# Single GPU
python recipes/llm/finetune.py --config examples/llm/llama_3_2_1b_squad.yaml

# Multi-GPU with torchrun
torchrun --nproc-per-node=2 recipes/llm/finetune.py --config examples/llm/llama_3_2_1b_squad.yaml
```

## Understand the Configuration

The example uses a YAML configuration file that defines all training parameters:

::::{dropdown} Key configuration sections
:icon: gear

```yaml
# Training schedule
step_scheduler:
  grad_acc_steps: 4
  num_epochs: 1
  ckpt_every_steps: 1000

# Model setup with NeMo optimizations
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B
  torch_dtype: torch.bfloat16
  use_liger_kernel: true

# Distributed training strategy
distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: none  # Auto-detect
  tp_size: 1

# Dataset configuration
dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  dataset_name: rajpurkar/squad
  split: train

# Optimizer settings
optimizer:
  _target_: torch.optim.Adam
  lr: 1.0e-5
  weight_decay: 0
```

::::



::::{dropdown} Migration to Megatron-Core (Advanced)
:icon: rocket

For maximum throughput, you can switch to Megatron-Core with minimal code changes:

```python
# Model class change
# Instead of: model=llm.HFAutoModelForCausalLM(model_id)
model = llm.LlamaModel(Llama32Config1B())

# Optimizer module change  
# Instead of: optim=fdl.build(llm.adam.pytorch_adam_with_flat_lr(lr=1e-5))
optim = MegatronOptimizerModule(config=opt_config)

# Trainer strategy change
# Instead of: strategy="fsdp2"
trainer = nl.Trainer(
    strategy=nl.MegatronStrategy(ddp="pytorch"),
    # ... other params
)
```

This enables optimal performance for training and post-training with minimal overhead.
::::

::::{dropdown} Lower-Level Component Usage
:icon: tools

For more granular control, you can use individual components:

```python
import torch
from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components.datasets.llm.squad import make_squad_dataset
from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager
from nemo_automodel.components._peft.lora import PeftConfig

# Load model with NeMo optimizations
model = NeMoAutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    torch_dtype=torch.bfloat16,
    use_liger_kernel=True,
    attn_implementation="flash_attention_2"
)

# Setup PEFT for memory-efficient training (optional)
peft_config = PeftConfig(
    dim=8,
    alpha=32,
    include_modules=["*.q_proj", "*.v_proj", "*.k_proj", "*.o_proj"]
)
model = peft_config.apply(model)

# Configure distributed training
dist_manager = FSDP2Manager(dp_size=None, tp_size=1)
model = dist_manager.wrap_model(model)

# Load dataset
dataset = make_squad_dataset("rajpurkar/squad", "train")

# Setup optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Training loop (simplified)
for batch in dataset:
    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
```

::::

## Try Different Models and Tasks

### Vision Language Models

Fine-tune multimodal models for image understanding:

```bash
automodel finetune vlm -c examples/vlm/gemma_3_vl_4b_medpix.yaml
```

### PEFT with LoRA

Memory-efficient training with LoRA adapters:

```bash
automodel finetune llm -c examples/llm/llama_3_2_1b_hellaswag_peft.yaml
```

### Different Model Architectures

Try other supported models:

```bash
# Qwen models
automodel finetune llm -c examples/llm/qwen_3_0p6b_hellaswag.yaml

# Or use any Hugging Face model by modifying the config:
# pretrained_model_name_or_path: microsoft/DialoGPT-medium
```

## What Happens During Training

When you run the command, NeMo Automodel:

1. **Downloads model**: Fetches Llama 3.2 1B from Hugging Face Hub
2. **Loads dataset**: Downloads and preprocesses SQuAD dataset
3. **Applies optimizations**: Enables Liger kernels and Flash Attention 2
4. **Configures distributed training**: Sets up FSDP2 for multi-GPU
5. **Starts training**: Runs supervised fine-tuning with gradient accumulation
6. **Saves checkpoints**: Periodic saves during training
7. **Exports model**: Final model ready for inference

## Expected Output

You should observe output like:

```console
INFO: Loading model meta-llama/Llama-3.2-1B...
INFO: Applying Liger kernel optimizations...
INFO: Loading SQuAD dataset...
INFO: Starting training with FSDP2...
Epoch 1/1: 100%|████████| 1000/1000 [05:23<00:00, 3.09it/s, loss=0.82]
INFO: Training completed. Model saved to checkpoints/
```

## Next Steps

Now that you've run your first training job:

1. **Explore configurations**: Refer to the [YAML Configuration Reference](../references/yaml-configuration-reference.md) for all options
2. **Try advanced features**: Learn about [Slurm Integration](../guides/launcher/slurm.md) for cluster training
3. **Understand the architecture**: Read the [Architecture Overview](../about/architecture-overview.md)
4. **Scale up**: Try larger models and datasets
5. **Get help**: Check the [Troubleshooting Reference](../references/troubleshooting-reference.md) for common issues

## Advanced: Extending AutoModel

NeMo AutoModel currently supports the `AutoModelForCausalLM` class for text generation. To add support for other tasks:

::::{dropdown} Adding New AutoModel Classes
:icon: plus

```python
# Create a subclass similar to HFAutoModelForCausalLM
class HFAutoModelForSeq2SeqLM(BaseAutoModel):
    def __init__(self, model_id, **kwargs):
        # Adapt initializer for your specific use case
        super().__init__(model_id, **kwargs)
    
    def training_step(self, batch, batch_idx):
        # Implement training logic for sequence-to-sequence
        pass
    
    def validation_step(self, batch, batch_idx):
        # Implement validation logic
        pass
    
    def configure_model(self):
        # Model configuration for your task
        pass
    
    def save_checkpoint(self, filepath):
        # Custom checkpoint handling
        pass
    
    def load_checkpoint(self, filepath):
        # Custom checkpoint loading
        pass
```

You'll also need to:

1. Implement appropriate checkpoint handling
2. Create a new data module with custom batch preprocessing
3. Adapt training/validation steps for your specific use case

Refer to the existing `HFAutoModelForCausalLM` class as a reference implementation.

::::

## Learn More

- {doc}`../about/index` - Platform overview and key features
- {doc}`../guides/llm/sft` - Deep dive into LLM fine-tuning
- {doc}`../guides/vlm/index` - Vision-language model training
- {doc}`../model-coverage/index` - Supported model architectures
- [NeMo Framework GitHub](https://github.com/NVIDIA/NeMo) - Full reference examples and comprehensive documentation
