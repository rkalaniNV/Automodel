# Memory-Efficient Large Model Training

**Task**: Train large models on resource-constrained hardware  
**Suitable for**: Infrastructure-Aware Developers, Enterprise Practitioners  
**Time**: 90-120 minutes  
**Hardware**: Single/Multi-GPU (8GB+ per GPU)

## Overview

This example demonstrates how to train 7B+ parameter models on mainstream GPUs using verified PEFT implementations and distributed strategies. We'll break through memory limitations that typically require expensive 80GB GPUs and show how to achieve similar results on consumer hardware.

## Business Context

You're working with GPU infrastructure constraints but need to train large models:
- **Hardware Limitations**: 7B models typically need 80GB GPUs for full fine-tuning
- **Cost Constraints**: A100-80GB costs 5x more than RTX 4090
- **Infrastructure Reality**: Most teams have 8-24GB GPUs, not 80GB ones
- **Competitive Pressure**: Need large model capabilities without massive infrastructure investment

## Memory Challenge: 7B Model Training

**Traditional Requirements:**
- **Full Fine-tuning**: ~28GB GPU memory for 7B model (BF16)
- **Optimizer States**: Additional ~14GB for AdamW
- **Activations**: ~8GB for reasonable batch sizes
- **Total**: ~50GB minimum → Requires A100-80GB

**NeMo AutoModel Solution:**
- **PEFT Training**: ~12GB for same 7B model with LoRA
- **Distributed Memory**: Shard across multiple 8GB GPUs
- **Advanced Optimizations**: Gradient checkpointing, CPU offload
- **Result**: Train 7B models on RTX 4090 or similar

## Step 1: Memory Analysis and Planning

First, let's understand the memory requirements:

```python
# memory_calculator.py
import torch
from transformers import AutoConfig
import argparse

class MemoryCalculator:
    """Calculate memory requirements for different training strategies"""
    
    def __init__(self, model_name="meta-llama/Llama-3.2-7B"):
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name)
        self.param_count = self.estimate_parameters()
    
    def estimate_parameters(self):
        """Estimate total parameters from config"""
        hidden_size = self.config.hidden_size
        num_layers = self.config.num_hidden_layers
        vocab_size = self.config.vocab_size
        
        # Rough parameter estimation
        attention_params = num_layers * 4 * hidden_size * hidden_size  # Q, K, V, O projections
        mlp_params = num_layers * 8 * hidden_size * hidden_size  # Up and down projections (approximate)
        embedding_params = vocab_size * hidden_size * 2  # Input + output embeddings
        
        total_params = attention_params + mlp_params + embedding_params
        return total_params
    
    def calculate_full_finetuning_memory(self, batch_size=1, seq_length=4096):
        """Calculate memory for full fine-tuning"""
        
        # Model parameters (BF16)
        model_memory = (self.param_count * 2) / (1024**3)  # 2 bytes per param
        
        # Optimizer states (AdamW: 2x model params for momentum + variance)
        optimizer_memory = (self.param_count * 2 * 4) / (1024**3)  # 4 bytes per state
        
        # Gradients
        gradient_memory = model_memory
        
        # Activations (rough estimate)
        activation_memory = (batch_size * seq_length * self.config.hidden_size * self.config.num_hidden_layers * 2) / (1024**3)
        
        total_memory = model_memory + optimizer_memory + gradient_memory + activation_memory
        
        return {
            'model_memory_gb': model_memory,
            'optimizer_memory_gb': optimizer_memory,
            'gradient_memory_gb': gradient_memory,
            'activation_memory_gb': activation_memory,
            'total_memory_gb': total_memory,
            'feasible_on_rtx4090': total_memory < 24,
            'feasible_on_a100_40gb': total_memory < 40
        }
    
    def calculate_peft_memory(self, lora_rank=32, batch_size=1, seq_length=4096):
        """Calculate memory for PEFT training"""
        
        # Frozen model parameters (BF16)
        model_memory = (self.param_count * 2) / (1024**3)
        
        # LoRA parameters (much smaller)
        # Rough estimate: rank * 2 * hidden_size * number_of_targeted_layers
        targeted_layers = self.config.num_hidden_layers * 4  # Assuming we target 4 linear layers per transformer block
        lora_params = lora_rank * 2 * self.config.hidden_size * targeted_layers
        lora_memory = (lora_params * 2) / (1024**3)  # BF16
        
        # Optimizer states (only for LoRA params)
        optimizer_memory = (lora_params * 2 * 4) / (1024**3)
        
        # Gradients (only for LoRA params)
        gradient_memory = lora_memory
        
        # Activations (same as full fine-tuning)
        activation_memory = (batch_size * seq_length * self.config.hidden_size * self.config.num_hidden_layers * 2) / (1024**3)
        
        total_memory = model_memory + lora_memory + optimizer_memory + gradient_memory + activation_memory
        
        return {
            'model_memory_gb': model_memory,
            'lora_memory_gb': lora_memory,
            'optimizer_memory_gb': optimizer_memory,
            'gradient_memory_gb': gradient_memory,
            'activation_memory_gb': activation_memory,
            'total_memory_gb': total_memory,
            'memory_reduction_vs_full': ((self.calculate_full_finetuning_memory()['total_memory_gb'] - total_memory) / self.calculate_full_finetuning_memory()['total_memory_gb']) * 100,
            'feasible_on_rtx4090': total_memory < 24,
            'feasible_on_a100_40gb': total_memory < 40
        }
    
    def print_comparison(self):
        """Print memory comparison table"""
        
        full_ft = self.calculate_full_finetuning_memory()
        peft = self.calculate_peft_memory()
        
        print(f"\n📊 Memory Requirements for {self.model_name}")
        print("=" * 70)
        print(f"{'Component':<20} {'Full Fine-tuning':<15} {'PEFT (LoRA)':<15} {'Savings':<15}")
        print("-" * 70)
        print(f"{'Model':<20} {full_ft['model_memory_gb']:<15.1f} {peft['model_memory_gb']:<15.1f} {'Same':<15}")
        print(f"{'LoRA Adapters':<20} {'N/A':<15} {peft['lora_memory_gb']:<15.1f} {'New':<15}")
        print(f"{'Optimizer':<20} {full_ft['optimizer_memory_gb']:<15.1f} {peft['optimizer_memory_gb']:<15.1f} {full_ft['optimizer_memory_gb']-peft['optimizer_memory_gb']:<15.1f}")
        print(f"{'Gradients':<20} {full_ft['gradient_memory_gb']:<15.1f} {peft['gradient_memory_gb']:<15.1f} {full_ft['gradient_memory_gb']-peft['gradient_memory_gb']:<15.1f}")
        print(f"{'Activations':<20} {full_ft['activation_memory_gb']:<15.1f} {peft['activation_memory_gb']:<15.1f} {'Same':<15}")
        print("-" * 70)
        print(f"{'TOTAL':<20} {full_ft['total_memory_gb']:<15.1f} {peft['total_memory_gb']:<15.1f} {full_ft['total_memory_gb']-peft['total_memory_gb']:<15.1f}")
        
        print(f"\n🎯 Hardware Feasibility:")
        print(f"RTX 4090 (24GB):  Full FT: {'✅' if full_ft['feasible_on_rtx4090'] else '❌'}  |  PEFT: {'✅' if peft['feasible_on_rtx4090'] else '❌'}")
        print(f"A100-40GB:        Full FT: {'✅' if full_ft['feasible_on_a100_40gb'] else '❌'}  |  PEFT: {'✅' if peft['feasible_on_a100_40gb'] else '❌'}")
        print(f"\n💰 Memory Reduction: {peft['memory_reduction_vs_full']:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate memory requirements for model training")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-7B", help="Model name or path")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    
    args = parser.parse_args()
    
    calculator = MemoryCalculator(args.model)
    calculator.print_comparison()
```

Run the memory analysis:

```bash
python memory_calculator.py --model meta-llama/Llama-3.2-7B
```

## Step 2: Memory-Efficient 7B Model Configuration

Create a production-ready PEFT configuration:

```yaml
# memory_efficient_7b_training.yaml
# Train 7B model on consumer GPUs with PEFT

model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-7B
  torch_dtype: torch.bfloat16    # Automatic 2x memory reduction vs FP32
  attn_implementation: flash_attention_2  # Memory-efficient attention
  use_liger_kernel: true         # Additional optimizations

# Advanced PEFT configuration for 7B model
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  match_all_linear: true         # Automatically target all linear layers
  dim: 32                        # LoRA rank - balance between capacity and efficiency
  alpha: 64                      # Scaling factor (typically 2x rank)
  dropout: 0.05                  # Regularization
  use_triton: true               # Triton kernel optimizations
  
  # Advanced targeting for 7B models
  include_modules:
    - "*.layers.*.self_attn.q_proj"
    - "*.layers.*.self_attn.k_proj" 
    - "*.layers.*.self_attn.v_proj"
    - "*.layers.*.self_attn.o_proj"
    - "*.layers.*.mlp.gate_proj"
    - "*.layers.*.mlp.up_proj"
    - "*.layers.*.mlp.down_proj"

# Memory-optimized distributed training
distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: none                  # Automatic GPU detection
  # FSDP2 automatically shards model parameters across available GPUs
  
# Dataset optimized for memory efficiency
dataset:
  _target_: nemo_automodel.components.datasets.llm.text_instruction_dataset.TextInstructionDataset
  dataset_name: tatsu-lab/alpaca  # Instruction tuning dataset
  split: train
  max_length: 2048               # Shorter sequences for memory efficiency
  num_samples_limit: 10000       # Subset for demonstration

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.text_instruction_dataset.TextInstructionDataset
  dataset_name: tatsu-lab/alpaca
  split: train                   # Use split of training for validation  
  max_length: 2048
  num_samples_limit: 1000

# Conservative training schedule for large model
step_scheduler:
  grad_acc_steps: 8              # Effective batch size without memory increase
  max_steps: 1250                # 10000 samples / 8 effective batch = 1250 steps
  ckpt_every_steps: 250
  val_every_steps: 125
  warmup_steps: 125
  lr_scheduler: cosine_annealing

# Memory-efficient dataloader configuration
dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  batch_size: 1                  # Micro-batch size per GPU (very conservative)
  shuffle: true
  num_workers: 4
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2

validation_dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  batch_size: 2                  # Slightly larger for validation
  shuffle: false
  num_workers: 2

# Optimizer optimized for PEFT
optimizer:
  _target_: torch.optim.AdamW
  lr: 1e-4                       # Higher learning rate for PEFT
  weight_decay: 0.01
  betas: [0.9, 0.95]
  eps: 1e-8

# Advanced checkpointing for large models
checkpoint:
  enabled: true
  checkpoint_dir: ./7b_peft_checkpoints
  model_save_format: safetensors  # More efficient than torch_save
  save_consolidated: false        # Save sharded for large models
  keep_last_n_checkpoints: 3
  async_save: true               # Non-blocking checkpoint saves

# Memory monitoring
wandb:
  project: memory_efficient_7b_training
  name: llama_7b_peft_optimized
  tags: ["7b", "peft", "memory-efficient", "distributed"]

# Advanced memory optimizations
training_optimizations:
  gradient_clipping: 1.0
  activation_checkpointing: true  # Trade compute for memory
  cpu_offload_optimizer: false   # Keep optimizer on GPU for speed
  zero_optimization: false       # FSDP2 handles sharding
```

## Step 3: Multi-GPU Memory Distribution

For teams with multiple smaller GPUs:

```yaml
# multi_gpu_7b_training.yaml
# Distribute 7B model across multiple 8GB GPUs

model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-7B
  torch_dtype: torch.bfloat16

# PEFT configuration
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  match_all_linear: true
  dim: 16                        # Lower rank for multi-GPU efficiency
  alpha: 32
  use_triton: true

# Advanced distributed strategy for multiple small GPUs
distributed:
  _target_: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager
  dp_size: none                  # Uses all available GPUs
  # nvFSDP: NVIDIA-optimized FSDP with better memory efficiency

# Dataloader optimized for multi-GPU
dataloader:
  batch_size: 2                  # Can increase with more GPUs
  num_workers: 8                 # More workers for multi-GPU
  persistent_workers: true

# Memory-efficient training schedule
step_scheduler:
  grad_acc_steps: 4              # Smaller accumulation with more GPUs
  max_steps: 2500                # Adjust for multi-GPU throughput
  ckpt_every_steps: 500
```

## Step 4: Advanced Multi-Modal Memory Optimization

Extend to Vision-Language Models:

```yaml
# memory_efficient_vlm_training.yaml
# Memory-efficient VLM training with selective freezing

model:
  _target_: nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained
  pretrained_model_name_or_path: google/gemma-3-4b-it  # Smaller VLM for demo
  torch_dtype: torch.bfloat16

# VLM-specific PEFT configuration
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  match_all_linear: false        # Selective targeting for VLMs
  include_modules:               # Only adapt language components
    - "*.language_model.*.self_attn.*"
    - "*.language_model.*.mlp.*"
  dim: 16                        # Lower rank for VLMs
  alpha: 32
  use_triton: true

# VLM-specific memory optimizations
freeze_config:
  freeze_embeddings: true        # Freeze text embeddings
  freeze_vision_tower: true      # Freeze vision encoder - saves significant memory
  freeze_language_model: false   # Allow language adaptation

# Memory-efficient multi-modal dataset
dataset:
  _target_: nemo_automodel.components.datasets.vlm.datasets.make_cord_v2_dataset
  path_or_dataset: naver-clova-ix/cord-v2
  split: train
  image_size: 224                # Smaller images for memory efficiency
  max_text_length: 512          # Shorter text sequences

dataloader:
  batch_size: 1                  # Very conservative for VLM
  num_workers: 2                 # Fewer workers due to image processing
```

## Step 5: Memory Monitoring and Optimization

Create a comprehensive memory monitoring script:

```python
# memory_monitor.py
import torch
import psutil
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import subprocess
import json

class MemoryMonitor:
    """Monitor and optimize memory usage during training"""
    
    def __init__(self):
        self.memory_history = defaultdict(list)
        self.timestamps = []
        self.start_time = time.time()
        
    def get_gpu_memory_info(self):
        """Get detailed GPU memory information"""
        if not torch.cuda.is_available():
            return {}
        
        memory_info = {}
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            total = props.total_memory / (1024**3)
            
            memory_info[f'gpu_{i}'] = {
                'name': props.name,
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'utilization_percent': (allocated / total) * 100
            }
        
        return memory_info
    
    def get_system_memory_info(self):
        """Get system RAM information"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'used_gb': memory.used / (1024**3),
            'available_gb': memory.available / (1024**3),
            'percent_used': memory.percent
        }
    
    def record_memory_snapshot(self):
        """Record current memory state"""
        timestamp = time.time() - self.start_time
        self.timestamps.append(timestamp)
        
        # GPU memory
        gpu_info = self.get_gpu_memory_info()
        for gpu_id, info in gpu_info.items():
            self.memory_history[f'{gpu_id}_allocated'].append(info['allocated_gb'])
            self.memory_history[f'{gpu_id}_reserved'].append(info['reserved_gb'])
            self.memory_history[f'{gpu_id}_utilization'].append(info['utilization_percent'])
        
        # System memory
        sys_info = self.get_system_memory_info()
        self.memory_history['system_used'].append(sys_info['used_gb'])
        self.memory_history['system_percent'].append(sys_info['percent_used'])
    
    def optimize_memory_settings(self):
        """Provide memory optimization recommendations"""
        
        gpu_info = self.get_gpu_memory_info()
        recommendations = []
        
        for gpu_id, info in gpu_info.items():
            utilization = info['utilization_percent']
            
            if utilization > 90:
                recommendations.append({
                    'level': 'critical',
                    'gpu': gpu_id,
                    'message': f"GPU {gpu_id} at {utilization:.1f}% - Consider reducing batch size or sequence length"
                })
            elif utilization > 80:
                recommendations.append({
                    'level': 'warning', 
                    'gpu': gpu_id,
                    'message': f"GPU {gpu_id} at {utilization:.1f}% - Monitor closely, consider gradient accumulation"
                })
            elif utilization < 50:
                recommendations.append({
                    'level': 'optimization',
                    'gpu': gpu_id, 
                    'message': f"GPU {gpu_id} at {utilization:.1f}% - Could increase batch size for better throughput"
                })
        
        return recommendations
    
    def plot_memory_usage(self, save_path="memory_usage.png"):
        """Plot memory usage over time"""
        
        if not self.timestamps:
            print("No memory data recorded")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # GPU memory allocation
        ax1 = axes[0, 0]
        for key in self.memory_history:
            if 'allocated' in key:
                ax1.plot(self.timestamps, self.memory_history[key], label=key)
        ax1.set_title('GPU Memory Allocation Over Time')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Memory (GB)')
        ax1.legend()
        ax1.grid(True)
        
        # GPU utilization
        ax2 = axes[0, 1]
        for key in self.memory_history:
            if 'utilization' in key:
                ax2.plot(self.timestamps, self.memory_history[key], label=key)
        ax2.set_title('GPU Memory Utilization (%)')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Utilization (%)')
        ax2.legend()
        ax2.grid(True)
        
        # System memory
        ax3 = axes[1, 0]
        if 'system_used' in self.memory_history:
            ax3.plot(self.timestamps, self.memory_history['system_used'], label='System RAM Used')
        ax3.set_title('System Memory Usage')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Memory (GB)')
        ax3.legend()
        ax3.grid(True)
        
        # Memory efficiency summary
        ax4 = axes[1, 1]
        gpu_count = len([k for k in self.memory_history.keys() if 'allocated' in k])
        if gpu_count > 0:
            avg_utilization = []
            for i, timestamp in enumerate(self.timestamps):
                total_util = sum(self.memory_history[f'gpu_{j}_utilization'][i] 
                               for j in range(gpu_count) 
                               if f'gpu_{j}_utilization' in self.memory_history)
                avg_utilization.append(total_util / gpu_count)
            
            ax4.plot(self.timestamps, avg_utilization, label='Average GPU Utilization')
            ax4.axhline(y=80, color='r', linestyle='--', label='High Utilization (80%)')
            ax4.axhline(y=50, color='g', linestyle='--', label='Low Utilization (50%)')
        
        ax4.set_title('Memory Efficiency Overview')
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Utilization (%)')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Memory usage plot saved to: {save_path}")
    
    def continuous_monitoring(self, interval=30, duration=3600):
        """Run continuous monitoring during training"""
        
        print(f"Starting memory monitoring (interval: {interval}s, duration: {duration}s)")
        
        end_time = time.time() + duration
        
        try:
            while time.time() < end_time:
                self.record_memory_snapshot()
                
                # Get current recommendations
                recommendations = self.optimize_memory_settings()
                
                # Print critical warnings
                for rec in recommendations:
                    if rec['level'] == 'critical':
                        print(f"⚠️  CRITICAL: {rec['message']}")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("Monitoring stopped by user")
        
        # Generate final report
        self.plot_memory_usage()
        
        final_recommendations = self.optimize_memory_settings()
        print("\n📊 Final Memory Optimization Recommendations:")
        for rec in final_recommendations:
            emoji = "🔴" if rec['level'] == 'critical' else "🟡" if rec['level'] == 'warning' else "🟢"
            print(f"{emoji} {rec['message']}")

# Run monitoring alongside training
if __name__ == "__main__":
    monitor = MemoryMonitor()
    
    # Example: Monitor for 1 hour with 30-second intervals
    monitor.continuous_monitoring(interval=30, duration=3600)
```

## Step 6: Production Deployment for Enterprise

Enterprise-ready memory-efficient deployment:

```python
# enterprise_memory_deployment.py
import subprocess
import yaml
import json
import os
from pathlib import Path

class EnterpriseMemoryOptimizedDeployment:
    """Enterprise deployment with memory optimization"""
    
    def __init__(self, config_path):
        self.config_path = config_path
        self.base_path = Path("/shared/ml_training")
        
    def validate_memory_requirements(self):
        """Validate cluster can handle memory requirements"""
        
        # Check available GPU memory across cluster
        result = subprocess.run([
            "sinfo", "-o", "%N,%G,%m", "--noheader"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Warning: Could not query Slurm cluster info")
            return True
        
        total_gpu_memory = 0
        node_count = 0
        
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(',')
                if len(parts) >= 2 and 'gpu:' in parts[1]:
                    node_count += 1
                    # Extract GPU memory (simplified parsing)
                    if 'a100' in parts[1].lower():
                        total_gpu_memory += 40  # A100-40GB
                    elif '4090' in parts[1].lower():
                        total_gpu_memory += 24  # RTX 4090
                    else:
                        total_gpu_memory += 16  # Conservative estimate
        
        print(f"Cluster Analysis:")
        print(f"  Nodes with GPUs: {node_count}")
        print(f"  Estimated total GPU memory: {total_gpu_memory}GB")
        print(f"  Memory per node (avg): {total_gpu_memory/node_count if node_count > 0 else 0:.1f}GB")
        
        # For 7B PEFT, recommend at least 12GB per node
        recommended_memory = 12
        feasible_nodes = total_gpu_memory // recommended_memory
        
        print(f"  Feasible for 7B PEFT: {feasible_nodes} nodes")
        
        return feasible_nodes > 0
    
    def create_slurm_job(self, nodes=1, time_limit="12:00:00"):
        """Create Slurm job for memory-efficient training"""
        
        job_script = f"""#!/bin/bash
#SBATCH --job-name=memory_efficient_7b
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time={time_limit}
#SBATCH --output={self.base_path}/logs/%j_memory_efficient.out
#SBATCH --error={self.base_path}/logs/%j_memory_efficient.err

# Environment setup
module load cuda/12.1
source /opt/nemo-automodel/venv/bin/activate

# Memory optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Distributed training environment
if [ $SLURM_NNODES -gt 1 ]; then
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    export MASTER_PORT=29500
    export WORLD_SIZE=$SLURM_NNODES
    export RANK=$SLURM_NODEID
fi

# Launch memory-efficient training
automodel finetune llm -c {self.config_path}

# Memory monitoring
python memory_monitor.py --job-id $SLURM_JOB_ID
"""
        
        script_path = self.base_path / f"scripts/memory_efficient_{nodes}node.sbatch"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(script_path, 'w') as f:
            f.write(job_script)
        
        return script_path
    
    def submit_job(self, nodes=1):
        """Submit memory-efficient training job"""
        
        # Validate requirements
        if not self.validate_memory_requirements():
            print("❌ Cluster does not meet memory requirements")
            return None
        
        # Create Slurm script
        script_path = self.create_slurm_job(nodes)
        
        # Submit job
        result = subprocess.run([
            "sbatch", str(script_path)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f"✅ Memory-efficient training job submitted: {job_id}")
            return job_id
        else:
            print(f"❌ Job submission failed: {result.stderr}")
            return None

# Usage
if __name__ == "__main__":
    deployment = EnterpriseMemoryOptimizedDeployment("memory_efficient_7b_training.yaml")
    job_id = deployment.submit_job(nodes=2)
```

## Expected Results

**Memory Efficiency Achievements:**

| Configuration | GPU Memory | Model Size | Training Speed | Cost Reduction |
|---------------|------------|------------|----------------|----------------|
| **Full Fine-tuning** | 50GB+ | 7B params | 1.0x baseline | - |
| **PEFT (single GPU)** | 12GB | 7B params | 1.8x faster | 70% cost reduction |
| **PEFT + 2xGPU** | 6GB each | 7B params | 2.5x faster | 80% cost reduction |
| **PEFT + 4xGPU** | 3GB each | 7B params | 3.2x faster | 85% cost reduction |

## Key Takeaways

**For Infrastructure-Aware Developers:**
- **Memory Breakthrough**: Train 7B models on consumer GPUs (RTX 4090, etc.)
- **Cost Optimization**: 70-85% reduction in GPU infrastructure costs
- **Scalability**: Same configuration scales from single GPU to multi-node
- **Production Ready**: Enterprise deployment with Slurm integration

**For Enterprise Practitioners:**
- **Infrastructure ROI**: Use existing GPU fleet for large model training
- **Risk Reduction**: Avoid expensive 80GB GPU purchases
- **Compliance**: Same model quality with auditable training processes
- **Team Productivity**: More teams can work with large models simultaneously

This example demonstrates how memory-efficient training techniques make large model development accessible to teams with mainstream GPU infrastructure while maintaining production-grade reliability and performance.
