---
description: "Train 7B models on 8GB GPUs using Parameter-Efficient Fine-Tuning combined with distributed training strategies."
categories: ["model-training"]
tags: ["peft", "lora", "distributed-training", "memory-optimization", "large-scale", "gpu-accelerated"]
personas: ["mle-focused", "researcher-focused"]
difficulty: "intermediate"
content_type: "tutorial"
modality: "llm"
---

(tutorial-peft-distributed)=
# Train 7B Models on 8GB GPUs with PEFT + Distributed

Breakthrough memory limitations and train large models efficiently with Parameter-Efficient Fine-Tuning and distributed strategies.

:::{note}
**Difficulty Level**: Intermediate  
**Estimated Time**: 45-60 minutes  
**Persona**: Infrastructure-Aware AI Developers with GPU clusters or multi-GPU setups
:::

(tutorial-peft-prerequisites)=
## Prerequisites

- Completed {doc}`first-fine-tuning` for performance optimization basics
- Access to GPU(s) with 8GB+ memory (single GPU or multi-GPU setup)
- Experience with larger model training challenges

(tutorial-peft-learning-objectives)=
## What You'll Learn

Deploy advanced memory optimization techniques for production-scale training:

- **Memory Breakthrough**: Train 7B+ models on consumer GPUs through PEFT
- **Distributed Efficiency**: Scale training across multiple GPUs with FSDP2/nvFSDP
- **Production PEFT**: LoRA configuration for real-world deployment scenarios
- **Multi-Modal Scaling**: Extend techniques to Vision-Language Models
- **Performance Analysis**: Measure memory savings and training efficiency

(tutorial-peft-memory-challenge)=
## The Large Model Memory Challenge

**Infrastructure Reality Check:**

Most teams face memory constraints when scaling to production-size models:

| Model Size | Full Fine-Tuning Memory | PEFT Memory | Memory Savings |
|------------|------------------------|-------------|----------------|
| **7B params** | ~28GB GPU memory | ~12GB GPU memory | **57% reduction** |
| **13B params** | ~52GB GPU memory | ~18GB GPU memory | **65% reduction** |
| **70B params** | ~280GB GPU memory | ~45GB GPU memory | **84% reduction** |

**What This Means for Your Infrastructure:**
- **7B models**: Train on single RTX 4090/A100 instead of requiring 80GB GPUs
- **13B models**: Multi-GPU accessible on mainstream hardware  
- **70B models**: Achievable on standard 8xA100 clusters

**PEFT + Distributed Strategy:**
Combine Parameter-Efficient Fine-Tuning with distributed training to scale efficiently across your existing GPU infrastructure.

(tutorial-peft-step1-memory-efficient)=
## Step 1: Memory-Efficient 7B Model Training

Let's train a 7B model on consumer hardware using PEFT:

```bash
# Check if you can train 7B models with current setup
cd examples/llm
cat llama_3_2_7b_peft.yaml  # If available, or use the following config
```

**Production PEFT Configuration for 7B Model:**

::::{tab-set}
::: {tab-item} LLM
```yaml
# Memory-efficient 7B model training (LLM)
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-7B
  torch_dtype: torch.bfloat16    # Automatic 2x memory reduction

# Advanced PEFT configuration
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  match_all_linear: true         # Automatic module targeting
  dim: 32                        # LoRA rank (balance capacity/efficiency)
  alpha: 64                      # Scaling factor
  use_triton: true              # Hardware optimization
  dropout: 0.05                 # Regularization

# Memory-optimized distributed training
distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: none                 # Automatic GPU detection

# Memory-efficient data loading
dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  batch_size: 2                 # Conservative for 7B model
  num_workers: 4
  pin_memory: true
```
:::
::: {tab-item} VLM
```yaml
# Memory-efficient VLM training with PEFT (VLM)
model:
  _target_: nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained
  pretrained_model_name_or_path: google/gemma-3-4b-it
  torch_dtype: torch.bfloat16

peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  match_all_linear: false
  include_modules:
    - "*.language_model.*.self_attn.*"
    - "*.language_model.*.mlp.*"
  dim: 16
  alpha: 32
  use_triton: true

freeze_config:
  freeze_embeddings: true
  freeze_vision_tower: true
  freeze_language_model: false

dataset:
  _target_: nemo_automodel.components.datasets.vlm.datasets.make_cord_v2_dataset
  path_or_dataset: naver-clova-ix/cord-v2
  split: train

dataloader:
  batch_size: 1
  num_workers: 2
```
:::
::::

**Memory Breakdown:**
- **Base 7B model**: ~14GB (BF16)
- **PEFT parameters**: ~67MB (0.5% of model)
- **Optimizer states**: ~6GB
- **Activations**: ~4GB
- **Total**: ~12GB (fits on RTX 4090!)

(tutorial-peft-step2-multi-gpu)=
## Step 2: Multi-GPU Distributed PEFT

Scale PEFT training across multiple GPUs for even larger models:

::::{tab-set}
::: {tab-item} LLM
```bash
# Automatically distributes across all available GPUs (LLM)
automodel finetune llm -c llama_3_2_7b_peft.yaml
```
:::
::: {tab-item} VLM
```bash
# Automatically distributes across all available GPUs (VLM)
automodel finetune vlm -c memory_efficient_vlm_training.yaml
```
:::
::::

**Advanced Distributed Configuration:**

```yaml
# Multi-GPU PEFT for large models
distributed:
  _target_: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager
  dp_size: none                 # Uses all available GPUs
  # nvFSDP: NVIDIA-optimized FSDP with better memory efficiency

# Alternative: FSDP2 for maximum compatibility
distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: none
  tp_size: 1                    # Can enable tensor parallelism for very large models
  cp_size: 1

# Memory optimizations
step_scheduler:
  grad_acc_steps: 8             # Effective batch size without memory increase
  ckpt_every_steps: 500
  val_every_steps: 100

# Advanced checkpointing for large models
checkpoint:
  enabled: true
  checkpoint_dir: ./peft_checkpoints
  model_save_format: safetensors # More efficient than torch_save
  save_consolidated: false       # Save sharded for large models
```

(tutorial-peft-step3-monitoring)=
## Step 3: Monitor Memory-Efficient Training

::::{tab-set}
::: {tab-item} LLM
```bash
# Launch training with memory monitoring (LLM)
automodel finetune llm -c llama_3_2_7b_peft.yaml
```
:::
::: {tab-item} VLM
```bash
# Launch training with memory monitoring (VLM)
automodel finetune vlm -c memory_efficient_vlm_training.yaml
```
:::
::::

**Training Output You'll See:**

```text
[INFO] Model loaded: meta-llama/Llama-3.2-7B (6.74B parameters)
[INFO] PEFT enabled: LoRA adapters (67M trainable parameters, 0.99% of total)
[INFO] Memory optimized: BF16 + FSDP2 sharding
[Step 50] Loss: 1.456 | GPU 0: 11.2GB/24GB | GPU 1: 11.1GB/24GB | Speed: 1.8 steps/sec
[Step 100] Loss: 1.234 | Trainable params: 67M/6.74B (0.99%)
```

(tutorial-peft-step4-multimodal)=
## Step 4: Advanced Multi-Modal PEFT

Extend memory efficiency to Vision-Language Models:

```yaml
# Memory-efficient VLM training with PEFT
model:
  _target_: nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained
  pretrained_model_name_or_path: google/gemma-3-4b-it
  torch_dtype: torch.bfloat16

# VLM-specific PEFT configuration
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  match_all_linear: false       # Selective targeting for VLMs
  include_modules:              # Target specific components
    - "*.language_model.*.self_attn.*"
    - "*.language_model.*.mlp.*"
  dim: 16                       # Lower rank for VLMs
  alpha: 32
  use_triton: true

# VLM-specific optimizations
freeze_config:
  freeze_embeddings: true       # Freeze text embeddings
  freeze_vision_tower: true     # Freeze vision encoder  
  freeze_language_model: false  # Allow language adaptation

# Multi-modal dataset
dataset:
  _target_: nemo_automodel.components.datasets.vlm.datasets.make_cord_v2_dataset
  path_or_dataset: naver-clova-ix/cord-v2
  split: train
```

**VLM Memory Efficiency:**
- **Gemma 3-4B VLM**: ~8GB with PEFT (vs ~16GB full fine-tuning)
- **Freeze strategy**: Only adapt language components, preserve vision
- **Selective targeting**: Apply LoRA only where needed

(tutorial-peft-step5-production)=
## Step 5: Production PEFT Deployment

```python
# Working with PEFT models in production
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class PEFTModelManager:
    def __init__(self, base_model_path, device="cuda"):
        # Load base model once
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, 
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.current_adapter = None
    
    def load_adapter(self, adapter_path):
        """Load specific LoRA adapter for task"""
        from peft import PeftModel
        
        if self.current_adapter:
            # Unload previous adapter
            self.model = self.base_model
        
        # Load new adapter
        self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
        self.current_adapter = adapter_path
        
    def generate(self, prompt, max_length=100):
        """Generate with current adapter"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Usage: Multiple specialized models from one base
manager = PEFTModelManager("meta-llama/Llama-3.2-7B")

# Switch between different domain adapters
manager.load_adapter("./finance_adapter")
finance_response = manager.generate("Analyze this financial report...")

manager.load_adapter("./medical_adapter") 
medical_response = manager.generate("Explain this medical condition...")
```

(tutorial-peft-step6-performance)=
## Step 6: Performance Analysis and Scaling Guidelines

**Memory and Performance Benchmarks:**

| Configuration | GPU Memory | Training Speed | Model Quality |
|---------------|------------|----------------|---------------|
| **7B Full Fine-tuning** | 28GB | 1.0x | 100% |
| **7B + LoRA (r=16)** | 12GB | 1.8x | 95% |
| **7B + LoRA (r=32)** | 13GB | 1.6x | 98% |
| **7B + LoRA + 2xGPU** | 6.5GB each | 2.8x | 98% |

**Scaling Strategy for Infrastructure Teams:**

```python
# Infrastructure planning for different model sizes
def calculate_peft_requirements(model_size_b, num_gpus, lora_rank=32):
    """Calculate memory requirements for PEFT training"""
    
    # Base memory requirements (BF16)
    model_memory = model_size_b * 2  # GB
    optimizer_memory = model_memory * 0.5
    lora_memory = (lora_rank * 2 * 4096 * 32) / (1024**3)  # Rough estimate
    
    total_memory = (model_memory + optimizer_memory + lora_memory) / num_gpus
    
    return {
        'memory_per_gpu_gb': total_memory,
        'recommended_gpu': 'A100-40GB' if total_memory > 24 else 'RTX-4090',
        'feasible': total_memory < 40
    }

# Example calculations
print("7B model:", calculate_peft_requirements(7, 1))
print("13B model:", calculate_peft_requirements(13, 2))
print("30B model:", calculate_peft_requirements(30, 4))
```

**Production Decision Framework:**

1. **<= 8GB GPU memory**: Use PEFT for any model > 3B parameters
2. **Multi-GPU setup**: Combine PEFT + distributed for maximum efficiency  
3. **Fast iteration needs**: PEFT adapters allow rapid domain switching
4. **Production serving**: Load base model once, swap adapters per request

(tutorial-peft-multinode)=
## Advanced Multi-Node PEFT

```yaml
# Multi-node PEFT configuration for very large models
distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: none                 # Automatic multi-node distribution
  tp_size: 1                    # Can enable for >70B models
  
# Multi-node specific optimizations  
step_scheduler:
  grad_acc_steps: 16            # Larger accumulation for distributed
  max_steps: 5000
  
# Slurm integration for cluster deployment
slurm:
  nodes: 4                      # Multi-node PEFT training
  ntasks_per_node: 8
  time: "12:00:00"
  container_image: "nvcr.io/nvidia/nemo:dev"
```

(tutorial-peft-optimization-checklist)=
## Infrastructure Optimization Checklist

**For Infrastructure-Aware AI Developers:**

- [ ] **Baseline measurement**: Document current model memory usage
- [ ] **PEFT implementation**: Target 50-80% memory reduction
- [ ] **Distributed scaling**: Test across available GPU infrastructure  
- [ ] **Multi-adapter serving**: Plan for production model switching
- [ ] **Monitoring setup**: Track memory efficiency and training speed
- [ ] **Cost analysis**: Calculate GPU hour savings from PEFT

**Real-World Success Metrics:**
- **Memory Efficiency**: Train 2-3x larger models on same hardware
- **Training Speed**: 1.5-2x faster due to fewer parameters
- **Cost Reduction**: 60-80% less GPU time for large model training
- **Deployment Flexibility**: Multiple specialized models from one base

(tutorial-peft-next-steps)=
## Next Steps

**Scale to Enterprise Infrastructure:**

1. **[Deploy Multi-Node Training](multi-gpu-training.md)** - Enterprise Slurm cluster integration
2. **[Advanced PEFT Techniques](../../guides/llm/peft.md)** - Deep dive into LoRA optimization
3. **[Checkpointing Guide](../../guides/checkpointing.md)** - Manage large model training state

**Apply in Practice:**

- **[Memory-Efficient Training Example](../examples/memory-efficient-training.md)** - Complete PEFT workflow with benchmarks
- **[Multi-Modal Fine-Tuning](../examples/multimodal-finetuning.md)** - Apply PEFT to vision-language models
- **[Enterprise Use Cases](../use-cases/ml-engineers.md)** - Production PEFT deployment patterns

**Related Concepts:**

- **{ref}`Understanding PEFT <parameter-efficient-fine-tuning>`** - Technical background
- **[Model Support](../../model-coverage/llm.md)** - PEFT compatibility across architectures

**API Reference:**

- **[PEFT Components](../../api-docs/_peft/_peft.md)** - LoRA implementation and configuration
- **[Distributed Training](../../api-docs/distributed/distributed.md)** - Multi-GPU PEFT strategies
- **[Memory Optimization](../../api-docs/utils/utils.md)** - Memory management utilities

---

**Navigation:**
- ← {doc}`first-fine-tuning` Previous: Performance Optimization Basics
- ↑ {doc}`index` Back to Tutorials Overview  
- → {doc}`multi-gpu-training` Next: Production Cluster Deployment
