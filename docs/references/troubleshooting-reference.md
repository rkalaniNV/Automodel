---
description: "Comprehensive troubleshooting guide for NeMo Automodel with common errors, solutions, and debugging strategies"
tags: ["troubleshooting", "debugging", "errors", "solutions", "support"]
categories: ["reference"]
personas: ["mle-focused", "data-scientist-focused", "enterprise-focused"]
difficulty: "reference"
content_type: "troubleshooting"
modality: "universal"
---

(troubleshooting-reference)=
# Troubleshooting Reference

Comprehensive troubleshooting guide for NeMo Automodel covering common errors, solutions, and debugging strategies for AI developers.

## Overview

This reference provides systematic approaches to diagnosing and resolving issues encountered when using NeMo Automodel for training and fine-tuning workflows.

### Quick Diagnostic Checklist

1. **Environment**: Python version, package installations, GPU drivers
2. **Configuration**: YAML syntax, required parameters, target resolution
3. **Resources**: GPU memory, disk space, network connectivity
4. **Dependencies**: Optional packages, version compatibility

## Installation and Environment Issues

### Missing Dependencies

#### Core Dependencies Missing

**Error:**
```
ModuleNotFoundError: No module named 'nemo_automodel'
```

**Solution:**
```bash
# Install from PyPI
pip install nemo_automodel

# Or install from source
pip install -e .
```

#### Optional Dependencies

**Triton Missing:**
```
ImportError: triton is not installed. Please install it with `pip install triton`.
```

**Solution:**
```bash
pip install triton
```

**TorchAO Missing (for FP8 quantization):**
```
ImportError: torchao is not installed. Please install it with `pip install torchao`.
```

**Solution:**
```bash
pip install torchao
```

**Qwen VL Utils Missing:**
```
ImportError: qwen_vl_utils is not installed. Please install it with `pip install qwen-vl-utils`.
```

**Solution:**
```bash
pip install qwen-vl-utils
```

#### GPU Dependencies

**CUDA/NCCL Issues:**
```
RuntimeError: NCCL error in: ../torch/csrc/distributed/c10d/NCCLUtils.hpp:311
```

**Solutions:**
1. **Check CUDA compatibility:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   python -c "import torch; print(torch.version.cuda)"
   ```

2. **Verify NCCL installation:**
   ```bash
   python -c "import torch.distributed as dist; print('NCCL available:', dist.is_nccl_available())"
   ```

3. **Environment variables:**
   ```bash
   export NCCL_DEBUG=INFO
   export CUDA_VISIBLE_DEVICES=0,1,2,3
   ```

### Version Compatibility

**PyTorch Version Conflicts:**
```
AttributeError: module 'torch' has no attribute 'sdpa_kernel'
```

**Solution:**
Ensure PyTorch 2.0+ is installed:
```bash
pip install "torch>=2.0.0"
```

## Configuration Errors

### YAML Syntax Issues

#### Missing `_target_` Parameter

**Error:**
```
TypeError: 'NoneType' object is not callable
```

**Problematic Configuration:**
```yaml
# ❌ Missing _target_
model:
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B
```

**Solution:**
```yaml
# ✅ Include _target_
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B
```

#### Invalid Target Resolution

**Error:**
```
AttributeError: module 'nemo_automodel' has no attribute 'InvalidClass'
```

**Solution:**
Verify the correct import path:
```yaml
# ✅ Correct target paths
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained

distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager

loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy
```

#### Required Parameter Missing

**Error:**
```
TypeError: from_pretrained() missing 1 required positional argument: 'pretrained_model_name_or_path'
```

**Solution:**
Include all required parameters:
```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B  # Required
```

### Configuration File Issues

**File Not Found:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'config.yaml'
```

**Solutions:**
1. Check file path and working directory
2. Use absolute paths when necessary
3. Verify file permissions

## Model Loading Issues

### Hugging Face Model Access

#### Gated Model Access

**Error:**
```
HTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/meta-llama/Llama-3.2-1B
```

**Solution:**
Set up Hugging Face authentication:
```bash
# Install huggingface-hub
pip install huggingface-hub

# Login interactively
huggingface-cli login

# Or set token environment variable
export HF_TOKEN=your_token_here
```

#### Model Not Found

**Error:**
```
HTTPError: 404 Client Error: Not Found for url: https://huggingface.co/invalid/model-name
```

**Solution:**
Verify model name exists on Hugging Face Hub or check local path.

### Memory Issues

#### GPU Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 23.70 GiB total capacity)
```

**Solutions:**

1. **Reduce batch size:**
   ```yaml
   dataloader:
     batch_size: 1  # Reduce from higher value
   ```

2. **Increase gradient accumulation:**
   ```yaml
   step_scheduler:
     grad_acc_steps: 8  # Increase to maintain effective batch size
   ```

3. **Use gradient checkpointing:**
   ```yaml
   model:
     _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
     pretrained_model_name_or_path: meta-llama/Llama-3.2-1B
     gradient_checkpointing: true
   ```

4. **Enable CPU offloading (FSDP2):**
   ```yaml
   distributed:
     _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
     cpu_offload: true
   ```

5. **Use PEFT instead of full fine-tuning:**
   ```yaml
   peft:
     _target_: nemo_automodel.components._peft.lora.PeftConfig
     dim: 8
     alpha: 32
   ```

#### Model Too Large for GPU

**Error:**
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu
```

**Solutions:**

1. **Use model sharding:**
   ```yaml
   distributed:
     _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
     dp_size: 4  # Shard across 4 GPUs
   ```

2. **Mixed precision training:**
   ```yaml
   model:
     torch_dtype: torch.bfloat16  # or torch.float16
   ```

## Kernel and Optimization Issues

### Liger Kernel Failures

**Error:**
```
RuntimeError: Failed to apply Liger kernel patches
```

**Solution:**
Disable Liger kernels:
```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B
  use_liger_kernel: false
```

### SDPA Attention Issues

**Error:**
```
RuntimeError: No available kernel. Aborting execution.
```

**Solutions:**

1. **Fallback to eager attention:**
   ```yaml
   model:
     attn_implementation: eager
     use_sdpa_patching: false
   ```

2. **Specific SDPA backend:**
   ```yaml
   model:
     attn_implementation: sdpa
     sdpa_method: ["math"]  # or ["flash", "memory_efficient"]
   ```

### Flash Attention Issues

**Error:**
```
ValueError: FlashAttention only supports Ampere GPUs or newer.
```

**Solution:**
Use compatible attention implementation:
```yaml
model:
  attn_implementation: eager  # For older GPUs
```

## Distributed Training Issues

### Multi-GPU Setup Problems

#### NCCL Initialization Failure

**Error:**
```
RuntimeError: NCCL error in: ../torch/csrc/distributed/c10d/NCCLUtils.hpp:311, unhandled system error
```

**Solutions:**

1. **Check GPU topology:**
   ```bash
   nvidia-smi topo -m
   ```

2. **Set NCCL debug level:**
   ```bash
   export NCCL_DEBUG=INFO
   export NCCL_TREE_THRESHOLD=0
   ```

3. **Use DDP instead of FSDP2:**
   ```yaml
   distributed:
     _target_: nemo_automodel.components.distributed.ddp.DDPManager
   ```

#### Process Group Timeout

**Error:**
```
RuntimeError: Timed out initializing process group in store based barrier
```

**Solution:**
Increase timeout:
```yaml
dist_env:
  backend: nccl
  timeout_minutes: 30  # Increase from default
```

### Slurm Integration Issues

#### Container Issues

**Error:**
```
srun: error: failed to launch task
```

**Solutions:**

1. **Check container image availability:**
   ```yaml
   slurm:
     container_image: nvcr.io/nvidia/nemo:dev  # Verify image exists
   ```

2. **Verify mount points:**
   ```yaml
   slurm:
     extra_mounts: "/data:/workspace/data"  # Check source directory exists
   ```

3. **Set proper permissions:**
   ```bash
   chmod -R 755 /path/to/mount/directory
   ```

## Dataset and DataLoader Issues

### Data Loading Failures

#### Dataset Not Found

**Error:**
```
DatasetNotFoundError: Dataset rajpurkar/squad not found
```

**Solutions:**

1. **Check internet connectivity**
2. **Verify dataset name spelling**
3. **Use local dataset path:**
   ```yaml
   dataset:
     _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
     dataset_name: /path/to/local/dataset
   ```

#### Permission Errors

**Error:**
```
PermissionError: [Errno 13] Permission denied: '/root/.cache/huggingface'
```

**Solution:**
Set proper cache directory:
```bash
export HF_HOME=/writable/path/to/cache
export TRANSFORMERS_CACHE=/writable/path/to/cache
```

### Memory Issues in DataLoading

**Error:**
```
RuntimeError: DataLoader worker (pid 12345) is killed by signal: Killed.
```

**Solutions:**

1. **Reduce number of workers:**
   ```yaml
   dataloader:
     num_workers: 1  # Reduce from higher value
   ```

2. **Disable multiprocessing:**
   ```yaml
   dataloader:
     num_workers: 0
   ```

3. **Increase shared memory:**
   ```bash
   docker run --shm-size=8g ...
   ```

## Training Loop Issues

### Checkpoint Problems

#### Checkpoint Loading Failure

**Error:**
```
FileNotFoundError: Checkpoint not found at checkpoints/step_100
```

**Solutions:**

1. **Verify checkpoint directory:**
   ```yaml
   checkpoint:
     checkpoint_dir: ./checkpoints  # Ensure directory exists
   ```

2. **Check checkpoint format compatibility:**
   ```yaml
   checkpoint:
     model_save_format: safetensors  # or pytorch
   ```

#### Permission Issues

**Error:**
```
PermissionError: [Errno 13] Permission denied: 'checkpoints/'
```

**Solution:**
```bash
mkdir -p checkpoints
chmod 755 checkpoints
```

### NaN Loss Issues

**Error:**
```
RuntimeError: Loss became NaN during training
```

**Solutions:**

1. **Reduce learning rate:**
   ```yaml
   optimizer:
     lr: 1e-6  # Reduce from higher value
   ```

2. **Enable gradient clipping:**
   ```yaml
   training:
     gradient_clip_val: 1.0
   ```

3. **Check for mixed precision issues:**
   ```yaml
   model:
     torch_dtype: torch.bfloat16  # Instead of float16
   ```

## Performance and Optimization Issues

### Slow Training

#### Inefficient Configuration

**Solutions:**

1. **Enable optimizations:**
   ```yaml
   model:
     use_liger_kernel: true
     attn_implementation: flash_attention_2
   ```

2. **Optimize data loading:**
   ```yaml
   dataloader:
     num_workers: 4
     pin_memory: true
     persistent_workers: true
   ```

3. **Use appropriate batch size:**
   ```yaml
   dataloader:
     batch_size: 8  # Maximize GPU utilization
   
   step_scheduler:
     grad_acc_steps: 4  # Adjust for effective batch size
   ```

### Memory Leaks

**Symptoms:**
- Gradually increasing GPU memory usage
- Out of memory errors after many steps

**Solutions:**

1. **Clear cache periodically:**
   ```python
   if step % 100 == 0:
       torch.cuda.empty_cache()
   ```

2. **Check for reference cycles in custom code**

3. **Disable persistent workers if issues persist:**
   ```yaml
   dataloader:
     persistent_workers: false
   ```

## Debugging Strategies

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# For distributed training
import os
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
```

### Memory Profiling

```python
import torch

# Track memory usage
torch.cuda.memory_summary()

# Profile memory allocations
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    # Training code here
    pass

print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
```

### Configuration Validation

```python
from nemo_automodel.shared.config import ConfigNode

# Validate configuration before training
config = ConfigNode.from_yaml("config.yaml")
print("Configuration loaded successfully")

# Test component instantiation
try:
    model = config.model.build()
    print("Model instantiation successful")
except Exception as e:
    print(f"Model instantiation failed: {e}")
```

## Getting Additional Help

### Community Resources

1. **GitHub Issues**: Report bugs and feature requests
2. **Documentation**: Check latest documentation for updates
3. **Examples**: Review working configuration examples

### Debug Information Collection

When reporting issues, include:

1. **Environment info:**
   ```bash
   python --version
   pip list | grep -E "(torch|nemo|transformers)"
   nvidia-smi
   ```

2. **Configuration file** (sanitized)

3. **Full error traceback**

4. **Minimal reproduction case**

### Professional Support

For enterprise users:
- NVIDIA Developer Support
- Professional Services engagement
- Training and consultation services

## See Also

- {doc}`cli-command-reference` - CLI usage and options
- {doc}`yaml-configuration-reference` - Configuration parameters
- {doc}`api-interfaces-reference` - Python API reference
- {doc}`../guides/launcher/slurm` - Distributed training setup
