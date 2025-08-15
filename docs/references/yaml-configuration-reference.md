---
description: "Complete YAML configuration schema reference for NeMo Automodel with all parameters, examples, and best practices"
tags: ["yaml", "configuration", "schema", "training", "model", "dataset"]
categories: ["reference"]
personas: ["mle-focused", "data-scientist-focused", "researcher-focused"]
difficulty: "reference"
content_type: "reference"
modality: "universal"
---

(yaml-configuration-reference)=
# YAML Configuration Reference

Comprehensive reference for NeMo Automodel YAML configuration files, covering all sections, parameters, and configuration patterns.

## Configuration Overview

NeMo Automodel uses a flexible YAML-based configuration system that supports:

- **Object instantiation** via `_target_` pattern
- **Hierarchical organization** with nested sections  
- **Dynamic resolution** of classes and functions
- **Environment-specific overrides** via command-line

### Basic Structure

```yaml
# Training schedule and checkpointing
step_scheduler:
  grad_acc_steps: 4
  max_steps: 1000

# Model configuration
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B

# Distributed training setup  
distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager

# Dataset and training loop
dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
```

## Core Configuration Sections

### Step Scheduler

Controls training duration, checkpointing frequency, and validation intervals.

```yaml
step_scheduler:
  grad_acc_steps: 4              # Gradient accumulation steps
  max_steps: 1000                # Maximum training steps (optional)
  num_epochs: 1                  # Number of training epochs (optional)  
  ckpt_every_steps: 100          # Checkpoint frequency
  val_every_steps: 50            # Validation frequency (in gradient steps)
```

**Key Parameters:**

```{list-table}
:header-rows: 1
:widths: 25 15 60

* - Parameter
  - Type
  - Description
* - `grad_acc_steps`
  - integer
  - Number of forward passes before backward pass
* - `max_steps`
  - integer
  - Total training steps (alternative to `num_epochs`)
* - `num_epochs`
  - integer
  - Number of full dataset passes (alternative to `max_steps`)
* - `ckpt_every_steps`
  - integer
  - Save checkpoint every N gradient steps
* - `val_every_steps`
  - integer
  - Run validation every N gradient steps
```

### Model Configuration

Defines the base model, optimizations, and architecture-specific settings.

#### Language Models (LLM)

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B
  torch_dtype: torch.bfloat16
  use_liger_kernel: true
  use_sdpa_patching: true
  attn_implementation: flash_attention_2
```

#### Vision Language Models (VLM)

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained
  pretrained_model_name_or_path: google/gemma-3-4b-it
  torch_dtype: torch.bfloat16
  use_liger_kernel: false
  attn_implementation: eager
```

**Model Parameters:**

```{list-table}
:header-rows: 1
:widths: 30 15 55

* - Parameter
  - Type
  - Description
* - `_target_`
  - string
  - Model class (required)
* - `pretrained_model_name_or_path`
  - string
  - Hugging Face model identifier or local path
* - `torch_dtype`
  - string/type
  - Precision (`torch.float16`, `torch.bfloat16`, `auto`)
* - `use_liger_kernel`
  - boolean
  - Enable Liger optimized attention kernels
* - `use_sdpa_patching`
  - boolean
  - Apply SDPA attention optimizations
* - `attn_implementation`
  - string
  - Attention backend (`flash_attention_2`, `eager`, `sdpa`)
```

### Distributed Training

Configure multi-GPU and multi-node training strategies.

#### FSDP2 (Recommended)

```yaml
distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: none                  # Auto-detect data parallel size
  dp_replicate_size: 1           # Replicated parameters
  tp_size: 1                     # Tensor parallel size
  cp_size: 1                     # Context parallel size  
  sequence_parallel: false       # Enable sequence parallelism
```

#### DDP (Simple Multi-GPU)

```yaml
distributed:
  _target_: nemo_automodel.components.distributed.ddp.DDPManager
```

#### nvFSDP (NVIDIA Optimized)

```yaml
distributed:
  _target_: nemo_automodel.components.distributed.nvfsdp.NvFSDPManager
  dp_size: none
  tp_size: 1
  cp_size: 1
```

**Distributed Parameters:**

```{list-table}
:header-rows: 1
:widths: 25 15 60

* - Parameter
  - Type
  - Description
* - `dp_size`
  - integer/none
  - Data parallel size (`none` for auto-detection)
* - `tp_size`
  - integer
  - Tensor parallel size (1 = disabled)
* - `cp_size`
  - integer
  - Context parallel size (1 = disabled)
* - `sequence_parallel`
  - boolean
  - Enable sequence-level parallelism
```

### PEFT Configuration

Parameter-Efficient Fine-Tuning with LoRA adapters.

```yaml
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  match_all_linear: false        # Target all linear layers
  include_modules:               # Specific modules to target
    - "*.q_proj"
    - "*.v_proj"
    - "*.k_proj"
    - "*.o_proj"
  exclude_modules:               # Modules to exclude
    - "*vision_tower*"
    - "*lm_head*"
  dim: 8                         # LoRA rank
  alpha: 32                      # LoRA alpha scaling
  dropout: 0.0                   # Dropout rate
  use_triton: true              # Use Triton kernels
```

**PEFT Parameters:**

```{list-table}
:header-rows: 1
:widths: 25 15 60

* - Parameter
  - Type
  - Description
* - `match_all_linear`
  - boolean
  - Apply LoRA to all linear layers
* - `include_modules`
  - list
  - Specific module patterns to include
* - `exclude_modules`
  - list
  - Module patterns to exclude
* - `dim`
  - integer
  - LoRA rank (lower = fewer parameters)
* - `alpha`
  - integer
  - LoRA scaling factor
* - `dropout`
  - float
  - Dropout rate for adapters
```

### Dataset Configuration

#### LLM Datasets

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  dataset_name: rajpurkar/squad
  split: train
  limit_dataset_samples: 1000    # Optional: limit dataset size

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  dataset_name: rajpurkar/squad
  split: validation
  limit_dataset_samples: 64
```

#### VLM Datasets

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.vlm.datasets.make_medpix_dataset
  path_or_dataset: mmoukouba/MedPix-VQA
  split: train[:1000]

validation_dataset:
  _target_: nemo_automodel.components.datasets.vlm.datasets.make_medpix_dataset
  path_or_dataset: mmoukouba/MedPix-VQA
  split: validation[:100]
```

### DataLoader Configuration

```yaml
dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  batch_size: 8
  shuffle: false
  num_workers: 1
  pin_memory: true
  collate_fn: nemo_automodel.components.datasets.utils.default_collater

validation_dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  batch_size: 8
  collate_fn:
    _target_: nemo_automodel.components.datasets.vlm.collate_fns.default_collate_fn
    start_of_response_token: "<start_of_turn>model\n"
```

### Optimization

#### Optimizers

```yaml
optimizer:
  _target_: torch.optim.Adam
  lr: 1e-5
  weight_decay: 0.01
  betas: [0.9, 0.95]
  eps: 1e-8
```

**Common Optimizers:**
- `torch.optim.Adam` - Standard Adam optimizer
- `torch.optim.AdamW` - Adam with weight decay
- `torch.optim.SGD` - Stochastic gradient descent

#### Learning Rate Schedulers

```yaml
lr_scheduler:
  lr_decay_style: cosine
  min_lr: 1e-6
  warmup_steps: 100
```

### Loss Functions

```yaml
loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy
```

**Available Loss Functions:**
- `nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy` - Masked cross-entropy
- `nemo_automodel.components.loss.chunked_ce.ChunkedCrossEntropy` - Memory-efficient chunked CE

### Environment Configuration

#### Distributed Environment

```yaml
dist_env:
  backend: nccl                  # Communication backend
  timeout_minutes: 10            # Timeout for distributed operations
```

#### Random Number Generation

```yaml
rng:
  _target_: nemo_automodel.components.training.rng.StatefulRNG
  seed: 42
  ranked: true                   # Different seeds per rank
```

### Checkpointing

```yaml
checkpoint:
  enabled: true
  checkpoint_dir: checkpoints/
  model_save_format: safetensors
  save_every_steps: 100
```

### Model Freezing (VLM)

```yaml
freeze_config:
  freeze_embeddings: true        # Freeze embedding layers
  freeze_vision_tower: true      # Freeze vision components
  freeze_language_model: false   # Keep language model trainable
```

### Logging and Monitoring

#### Weights & Biases

```yaml
wandb:
  project: my_project
  entity: my_team
  name: experiment_name
  save_dir: ./wandb_logs
```

#### Alternative Logger Format

```yaml
logger:
  wandb_project: my_project
  wandb_entity: my_team
  wandb_exp_name: experiment_name
  wandb_save_dir: ./wandb_logs
```

### Slurm Integration

```yaml
slurm:
  job_name: llm-finetune
  nodes: 2
  ntasks_per_node: 8
  time: "02:00:00"
  account: my_account
  partition: gpu
  container_image: "nvcr.io/nvidia/nemo:dev"
  gpus_per_node: 8
  master_port: 13742
  hf_home: ~/.cache/huggingface
  wandb_key: ${WANDB_API_KEY}
  hf_token: ${HF_TOKEN}
```

## Configuration Patterns

### The `_target_` Pattern

NeMo Automodel uses the `_target_` pattern for dynamic object instantiation:

```yaml
component:
  _target_: package.module.ClassName
  parameter1: value1
  parameter2: value2
```

This resolves to:
```python
from package.module import ClassName
component = ClassName(parameter1=value1, parameter2=value2)
```

### Environment Variable Substitution

Reference environment variables in YAML:

```yaml
wandb:
  api_key: ${WANDB_API_KEY}
  
slurm:
  hf_token: ${HF_TOKEN}
```

### Conditional Configuration

Use different configurations based on environment:

```yaml
# Development
model:
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B
  
# Production (larger model)
# model:
#   pretrained_model_name_or_path: meta-llama/Llama-3.2-8B
```

## Complete Configuration Examples

### LLM Fine-tuning with PEFT

```yaml
step_scheduler:
  grad_acc_steps: 4
  max_steps: 1000
  ckpt_every_steps: 100

model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B
  torch_dtype: torch.bfloat16

peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  dim: 8
  alpha: 32

distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  tp_size: 1

dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  dataset_name: rajpurkar/squad
  split: train

optimizer:
  _target_: torch.optim.Adam
  lr: 1e-5
```

### VLM Fine-tuning with Freezing

```yaml
step_scheduler:
  grad_acc_steps: 8
  max_steps: 500

model:
  _target_: nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained
  pretrained_model_name_or_path: google/gemma-3-4b-it
  torch_dtype: torch.bfloat16

freeze_config:
  freeze_vision_tower: true
  freeze_embeddings: true

dataset:
  _target_: nemo_automodel.components.datasets.vlm.datasets.make_medpix_dataset
  path_or_dataset: mmoukouba/MedPix-VQA
  split: train[:1000]

dataloader:
  batch_size: 1
  collate_fn:
    _target_: nemo_automodel.components.datasets.vlm.collate_fns.default_collate_fn
    start_of_response_token: "<start_of_turn>model\n"
```

## Validation and Troubleshooting

### Required Parameters

Each configuration must include:
- `model` section with `_target_` and model identifier
- `dataset` section with appropriate dataset loader
- Training schedule (`step_scheduler`)

### Common Validation Errors

**Missing `_target_`:**
```yaml
# ❌ Wrong
model:
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B

# ✅ Correct  
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B
```

**Invalid target resolution:**
```yaml
# ❌ Wrong (non-existent class)
model:
  _target_: nemo_automodel.InvalidClass

# ✅ Correct
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
```

## See Also

- {doc}`cli-command-reference` - CLI usage patterns
- {doc}`../guides/llm/sft` - LLM training examples
- {doc}`../guides/vlm/index` - VLM training examples
- {doc}`api-interfaces-reference` - Python API reference
