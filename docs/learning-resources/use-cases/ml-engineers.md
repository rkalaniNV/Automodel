---
description: "Training pipeline and distributed system use cases for Machine Learning Engineers focused on YAML recipes and pipeline optimization."
categories: ["model-training"]
tags: ["fine-tuning", "optimization", "training-loop", "python-api", "yaml-config", "recipes"]
personas: ["mle-focused", "enterprise-focused"]
difficulty: "intermediate"
content_type: "example"
modality: "universal"
---

# Machine Learning Engineers Use Cases

Training pipeline and distributed system use cases for Machine Learning Engineers focused on YAML recipes, distributed training, and pipeline optimization with NeMo AutoModel.

:::{note}
**Target Audience**: Machine Learning Engineers  
**Focus**: Training pipelines, distributed training setup, YAML recipes, checkpoint management, GPU optimization
:::

## Overview

As a Machine Learning Engineer, you need robust training pipelines, efficient distributed training configurations, and optimized GPU utilization. NeMo AutoModel provides comprehensive YAML recipes, distributed training frameworks, and checkpoint management systems for production ML workflows.

---

## Use Case 1: Production Training Pipelines with YAML Recipes

**Context**: Standardized, reproducible training pipelines with comprehensive configuration management.

### NeMo AutoModel Solution

**Production Training Configuration**
::::{tab-set}
::: {tab-item} LLM (PEFT)
```{dropdown} production_training_pipeline.yaml
:open:
```yaml
# production_training_pipeline.yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-3B
  torch_dtype: torch.bfloat16
  attn_implementation: flash_attention_2
  use_liger_kernel: true

peft:
  _target_: nemo_automodel.components._peft.lora.LoRA
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  r: 16
  alpha: 32
  dropout: 0.1

dataset:
  _target_: nemo_automodel.components.datasets.llm.text_instruction_dataset.TextInstructionDataset
  data_path: "${DATA_PATH}/train.jsonl"
  max_length: 2048

step_scheduler:
  grad_acc_steps: ${GRAD_ACC_STEPS:4}
  max_steps: ${MAX_STEPS:2000}
  val_every_steps: ${VAL_EVERY:200}
  ckpt_every_steps: ${CKPT_EVERY:500}

dataloader:
  batch_size: ${BATCH_SIZE:4}
  num_workers: ${NUM_WORKERS:8}
  pin_memory: true

optimizer:
  _target_: torch.optim.AdamW
  lr: ${LEARNING_RATE:5e-5}
  weight_decay: ${WEIGHT_DECAY:0.01}

checkpoint:
  enabled: true
  checkpoint_dir: "${CHECKPOINT_DIR:./checkpoints}"
  keep_last_n_checkpoints: 3

wandb:
  project: "${WANDB_PROJECT:production_training}"
  name: "${WANDB_RUN_NAME:llama_training}"
  tags: ["production", "llama", "peft"]
```
```
:::
::: {tab-item} LLM (Non-PEFT)
```{dropdown} production_training_pipeline_no_peft.yaml
:open:
```yaml
# production_training_pipeline_no_peft.yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-3B
  torch_dtype: torch.bfloat16
  attn_implementation: flash_attention_2
  use_liger_kernel: true

dataset:
  _target_: nemo_automodel.components.datasets.llm.text_instruction_dataset.TextInstructionDataset
  data_path: "${DATA_PATH}/train.jsonl"
  max_length: 2048

step_scheduler:
  grad_acc_steps: ${GRAD_ACC_STEPS:4}
  max_steps: ${MAX_STEPS:2000}
  val_every_steps: ${VAL_EVERY:200}
  ckpt_every_steps: ${CKPT_EVERY:500}

dataloader:
  batch_size: ${BATCH_SIZE:4}
  num_workers: ${NUM_WORKERS:8}
  pin_memory: true

optimizer:
  _target_: torch.optim.AdamW
  lr: ${LEARNING_RATE:5e-5}
  weight_decay: ${WEIGHT_DECAY:0.01}

checkpoint:
  enabled: true
  checkpoint_dir: "${CHECKPOINT_DIR:./checkpoints}"
  keep_last_n_checkpoints: 3

wandb:
  project: "${WANDB_PROJECT:production_training}"
  name: "${WANDB_RUN_NAME:llama_training}"
  tags: ["production", "llama"]
```
```
:::
::: {tab-item} VLM
```{dropdown} vlm_training_pipeline.yaml
:open:
```yaml
# vlm_training_pipeline.yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained
  pretrained_model_name_or_path: google/gemma-3-4b-it
  torch_dtype: torch.bfloat16

dataset:
  _target_: nemo_automodel.components.datasets.vlm.datasets.make_cord_v2_dataset
  path_or_dataset: naver-clova-ix/cord-v2
  split: train

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  batch_size: 1
  num_workers: 0
  pin_memory: true
  collate_fn:
    _target_: nemo_automodel.components.datasets.vlm.collate_fns.default_collate_fn
    start_of_response_token: "<start_of_turn>model\n"

step_scheduler:
  grad_acc_steps: 8
  max_steps: 1000
```
```
:::
::::

---

## Use Case 2: Distributed Training Setup & Multi-GPU Optimization

**Context**: Scale training across multiple GPUs with optimal resource utilization.

### NeMo AutoModel Solution

**Distributed Training Configuration**
::::{tab-set}
::: {tab-item} LLM (PEFT)
```{dropdown} distributed_training.yaml
:open:
```yaml
# distributed_training.yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-7B
  torch_dtype: torch.bfloat16

distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  sharding_strategy: "full_shard"
  mixed_precision: true
  forward_prefetch: true
  backward_prefetch: true

peft:
  _target_: nemo_automodel.components._peft.lora.LoRA
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  r: 32
  alpha: 64

dataloader:
  batch_size: 2  # Per-GPU batch size
  num_workers: 4
  sampler: "distributed"

step_scheduler:
  grad_acc_steps: 8
  max_steps: 5000
```
```
:::
::: {tab-item} LLM (Non-PEFT)
```{dropdown} distributed_training_no_peft.yaml
:open:
```yaml
# distributed_training_no_peft.yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-7B
  torch_dtype: torch.bfloat16

distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  sharding_strategy: "full_shard"
  mixed_precision: true
  forward_prefetch: true
  backward_prefetch: true

dataloader:
  batch_size: 2
  num_workers: 4
  sampler: "distributed"

step_scheduler:
  grad_acc_steps: 8
  max_steps: 5000
```
```
:::
::: {tab-item} VLM
```{dropdown} vlm_distributed_training.yaml
:open:
```yaml
# vlm_distributed_training.yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained
  pretrained_model_name_or_path: google/gemma-3-4b-it
  torch_dtype: torch.bfloat16

distributed:
  _target_: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager
  dp_size: none
  tp_size: 1
  cp_size: 1

dataloader:
  batch_size: 1
  num_workers: 2
  sampler: "distributed"

step_scheduler:
  grad_acc_steps: 8
  max_steps: 1000
```
```
:::
::::

---

## Use Case 3: Checkpoint Management & Model Deployment

**Context**: Comprehensive checkpoint management and automated deployment pipelines.

### NeMo AutoModel Solution

**Checkpoint Management**
```{dropdown} checkpoint_management.yaml
:open:
```yaml
# checkpoint_management.yaml
checkpoint_management:
  checkpoint_config:
    enabled: true
    save_interval_steps: 500
    keep_last_n_checkpoints: 5
    validate_on_save: true

  export_config:
    formats: ["safetensors", "pytorch"]
    optimization_level: "O2"

deployment:
  serving_config:
    framework: "vllm"
    max_batch_size: 32
    tensor_parallel_size: 2

model_registry:
  backend: "mlflow"
  experiment_name: "llama_training"
```
```

---

## Get Started for ML Engineers

### Prerequisites
- Experience with distributed training and GPU optimization
- YAML configuration management and pipeline development
- Understanding of model deployment and serving infrastructure

### Development Path
1. **Pipeline Development**: Create standardized YAML recipes
2. **Distributed Training**: Scale training across multiple GPUs
3. **Checkpoint Management**: Implement robust checkpointing
4. **Deployment Automation**: Build automated deployment pipelines

### Quick Start
```bash
# Launch distributed training
torchrun --nproc_per_node=4 automodel finetune llm -c distributed_training.yaml
```

### Resources
- [Tutorials](../tutorials/index.md)
- [Examples](../examples/index.md)
- [YAML configuration reference](../../references/yaml-configuration-reference.md)
- [Python API Reference](../../references/python-api-reference.md)
- [Troubleshooting Reference](../../references/troubleshooting-reference.md)

---

**Success Metrics for ML Engineers:**
- **Pipeline Reliability**: 99%+ training success rate with error recovery
- **Distributed Efficiency**: 90%+ scaling efficiency across GPUs
- **Deployment Automation**: Fully automated model deployment
- **Configuration Management**: Standardized YAML recipes across team
