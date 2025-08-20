---
description: "Deploy enterprise-grade distributed training across multiple nodes using Slurm job scheduling and NeMo AutoModel."
categories: ["infrastructure-operations"]
tags: ["slurm", "distributed-training", "multi-node", "launcher", "enterprise", "cluster-management"]
personas: ["admin-focused", "devops-focused", "enterprise-focused"]
difficulty: "advanced"
content_type: "tutorial"
modality: "universal"
---

(tutorial-multi-node-slurm)=
# Deploy Multi-Node Training on Your Slurm Cluster Today

Production-ready multi-node training with built-in Slurm integration, containerization, and enterprise monitoring.

:::{note}
**Difficulty Level**: Advanced  
**Estimated Time**: 60-90 minutes  
**Persona**: Enterprise AI Practitioners and Cluster Administrators managing production ML infrastructure
:::

(tutorial-multi-node-prerequisites)=
## Prerequisites

- Access to Slurm cluster or multi-node GPU environment
- Completed {doc}`first-fine-tuning` and {doc}`parameter-efficient-fine-tuning`
- System administrator access for container/Slurm configuration

(tutorial-multi-node-learning-objectives)=
## What You'll Learn

Deploy production-scale training infrastructure that works today:

- **Slurm Integration**: Built-in multi-node job submission and management
- **Container Deployment**: Production containerization with NVIDIA optimizations
- **Multi-Node Scaling**: Real distributed training across cluster nodes
- **Enterprise Monitoring**: Job tracking, resource utilization, and failure recovery
- **Production Workflows**: Automated deployment pipelines for ML teams

(tutorial-multi-node-enterprise-reality)=
## Enterprise Cluster Training Reality

**What Enterprise Teams Actually Need:**

Most enterprise AI teams have similar infrastructure challenges:
- **Slurm-managed clusters** with shared GPU resources
- **Multi-tenant environments** requiring job isolation
- **Compliance requirements** for training job auditing
- **Container orchestration** for reproducible environments
- **Cost accountability** for GPU resource usage

**NeMo AutoModel's Production Solution:**
- **Built-in Slurm integration** - no custom scripts needed
- **Container-first approach** - production reproducibility
- **Automatic multi-node scaling** - efficient resource utilization
- **Enterprise logging** - audit trails and monitoring

(tutorial-multi-node-step1-assessment)=
## Step 1: Slurm Cluster Assessment

First, verify your cluster is ready for distributed training:

```bash
# Check Slurm cluster status
sinfo
squeue
sacct --format=JobID,JobName,State,Time,NodeList

# Check available GPU partitions
sinfo -o "%P %G %F"

# Verify container support (Enroot/Pyxis or Singularity)
which enroot || which singularity
```

**Typical Enterprise Slurm Setup:**
- **GPU partitions**: `gpu`, `dgx`, `a100` with different SLAs
- **Container runtime**: Enroot/Pyxis for NVIDIA containers
- **Shared storage**: `/shared`, `/lustre`, or `/gpfs` mounted on all nodes
- **Module system**: Environment modules for software management

(tutorial-multi-node-step2-configuration)=
## Step 2: Configure Multi-Node Training

NeMo AutoModel provides built-in Slurm integration through YAML configuration:

```yaml
# Multi-node training configuration with Slurm
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-7B
  torch_dtype: torch.bfloat16

# Production distributed strategy
distributed:
  _target_: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager
  dp_size: none  # Automatic multi-node distribution

# Built-in Slurm integration
slurm:
  job_name: "llm_7b_training"
  nodes: 4                          # Request 4 nodes
  ntasks_per_node: 8                # 8 GPUs per DGX node
  time: "24:00:00"                  # 24-hour time limit
  account: "research_project"       # Slurm account for billing
  partition: "gpu"                  # GPU partition
  
  # Container configuration
  container_image: "nvcr.io/nvidia/nemo:dev"
  hf_home: "/shared/models/.cache/huggingface"
  
  # Environment and credentials
  wandb_key: "${WANDB_API_KEY}"
  hf_token: "${HF_TOKEN}"
  
  # Additional mount points
  extra_mounts:
    - "/shared/datasets:/data"
    - "/shared/results:/results"

# Multi-node optimized training
step_scheduler:
  grad_acc_steps: 16        # Large accumulation for multi-node
  max_steps: 10000
  ckpt_every_steps: 1000
  
# Checkpoint to shared storage
checkpoint:
  enabled: true
  checkpoint_dir: "/shared/results/checkpoints"
  model_save_format: safetensors
  save_consolidated: false  # Sharded saves for large models
```

(tutorial-multi-node-step3-launch)=
## Step 3: Launch Multi-Node Training

Submit the training job to your Slurm cluster:

```bash
# Submit multi-node training job
automodel finetune llm -c multi_node_training.yaml

# What happens automatically:
# 1. Generates Slurm batch script
# 2. Submits job to queue with sbatch
# 3. Returns job ID for monitoring
# 4. Sets up multi-node communication
# 5. Launches distributed training across nodes
```

(tutorial-multi-node-step4-monitoring)=
## Step 4: Monitor Multi-Node Training

Track your distributed training job across the cluster:

```bash
# Monitor job status
squeue -j <job_id>

# Check job details and resource allocation
scontrol show job <job_id>

# Monitor training logs in real-time
tail -f /shared/results/slurm_<job_name>_<job_id>.out

# Check node-specific logs
ssh <node_name> tail -f /tmp/training_node.log
```

**Training Output You'll See:**

```text
# Slurm job startup
[Node 0] MASTER_ADDR=dgx-01
[Node 0] WORLD_SIZE=32  # 4 nodes × 8 GPUs
[Node 1] Joining training at dgx-01:29500
[Node 2] Joining training at dgx-01:29500
[Node 3] Joining training at dgx-01:29500

# Distributed training progress
[Step 100] Node 0/GPU 0: Loss=1.245 | Mem=18.2GB/40GB
[Step 100] Node 1/GPU 0: Loss=1.245 | Mem=18.1GB/40GB  
[Step 100] Avg Speed: 2.4 steps/sec across 32 GPUs
[Step 100] Effective Batch Size: 512 (16 per GPU × 32 GPUs)

# Checkpoint synchronization
[Step 1000] Saving checkpoint to /shared/results/checkpoints/step_1000/
[Step 1000] Checkpoint saved across 4 nodes (sharded format)
```

(tutorial-multi-node-step5-advanced)=
## Step 5: Advanced Multi-Node Configurations

**For Large-Scale Enterprise Deployments:**

```yaml
# Advanced multi-node configuration for 70B+ models
slurm:
  job_name: "llama_70b_enterprise"
  nodes: 16                         # Scale to 16 nodes
  ntasks_per_node: 8
  time: "48:00:00"
  account: "enterprise_ai"
  partition: "large_jobs"
  
  # Advanced Slurm options
  exclusive: true                   # Dedicated node access
  mail_type: "FAIL,END"            # Email notifications
  
# Advanced distributed strategy
distributed:
  _target_: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager
  dp_size: none
  tp_size: 2                        # Tensor parallelism for very large models
  cp_size: 1

# Enterprise-grade checkpointing
checkpoint:
  enabled: true
  checkpoint_dir: "/shared/enterprise/checkpoints"
  save_consolidated: false
  async_save: true                  # Non-blocking saves
  keep_last_n_checkpoints: 3
  
# Multi-node data loading optimizations
dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  batch_size: 1                     # Per GPU batch size
  num_workers: 8                    # Parallel data loading per node
  persistent_workers: true          # Keep workers alive
  prefetch_factor: 4                # Buffer more batches
```

(tutorial-multi-node-step6-production)=
## Step 6: Production Deployment Patterns

**Automated Training Pipeline:**

```bash
# Create production deployment script
cat > submit_enterprise_training.sh << 'EOF'
#!/bin/bash
set -e

# Production environment setup
export WANDB_API_KEY=$(cat /shared/secrets/wandb_key)
export HF_TOKEN=$(cat /shared/secrets/hf_token)

# Submit job with error handling
JOB_ID=$(automodel finetune llm -c enterprise_config.yaml)

if [ $? -eq 0 ]; then
    echo "Training job submitted: $JOB_ID"
    echo "Monitor: squeue -j $JOB_ID"
    echo "Logs: tail -f /shared/results/slurm_*_${JOB_ID}.out"
    
    # Set up automated monitoring
    python /shared/scripts/monitor_training.py --job_id $JOB_ID &
else
    echo "Job submission failed"
    exit 1
fi
EOF

chmod +x submit_enterprise_training.sh
```

(tutorial-multi-node-step7-enterprise)=
## Step 7: Enterprise Monitoring and Management

**Production Job Monitoring Script:**

```python
# monitor_training.py - Enterprise training monitoring
import subprocess
import time
import json
import logging
from datetime import datetime
from pathlib import Path

class ClusterTrainingMonitor:
    def __init__(self, job_id, alert_webhook=None):
        self.job_id = job_id
        self.alert_webhook = alert_webhook
        self.start_time = time.time()
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'/shared/logs/training_monitor_{job_id}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def check_job_status(self):
        """Check Slurm job status and resource usage"""
        try:
            # Get job status
            result = subprocess.run(['scontrol', 'show', 'job', self.job_id], 
                                  capture_output=True, text=True)
            
            if 'JobState=RUNNING' not in result.stdout:
                return self._parse_job_completion(result.stdout)
            
            # Get resource usage
            result = subprocess.run(['sstat', '-j', self.job_id, '--format=JobID,AveCPU,AveRSS,MaxRSS'], 
                                  capture_output=True, text=True)
            
            return self._parse_resource_usage(result.stdout)
            
        except Exception as e:
            self.logger.error(f"Status check failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def check_training_progress(self):
        """Monitor training progress from logs"""
        try:
            log_pattern = f"/shared/results/slurm_*_{self.job_id}.out"
            import glob
            log_files = glob.glob(log_pattern)
            
            if not log_files:
                return {'status': 'no_logs'}
            
            latest_log = max(log_files, key=lambda x: Path(x).stat().st_mtime)
            
            with open(latest_log, 'r') as f:
                lines = f.readlines()
            
            # Parse recent training metrics
            recent_lines = lines[-50:]  # Last 50 lines
            metrics = {}
            
            for line in recent_lines:
                if '[Step' in line and 'Loss:' in line:
                    # Extract step number and loss
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.startswith('[Step'):
                            step = int(part.split('/')[0].replace('[Step', ''))
                            metrics['current_step'] = step
                        elif part == 'Loss:' and i + 1 < len(parts):
                            loss = float(parts[i + 1])
                            metrics['current_loss'] = loss
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Progress check failed: {e}")
            return {'status': 'error'}
    
    def send_alert(self, message, severity="info"):
        """Send alerts for important events"""
        alert = {
            'job_id': self.job_id,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'runtime_hours': (time.time() - self.start_time) / 3600
        }
        
        self.logger.info(f"ALERT [{severity}]: {message}")
        
        if self.alert_webhook:
            try:
                import requests
                requests.post(self.alert_webhook, json=alert, timeout=5)
            except Exception as e:
                self.logger.error(f"Alert sending failed: {e}")
    
    def run_monitoring(self, check_interval=300):
        """Run continuous monitoring"""
        self.logger.info(f"Starting monitoring for job {self.job_id}")
        
        while True:
            try:
                # Check job status
                job_status = self.check_job_status()
                
                if job_status.get('status') in ['COMPLETED', 'FAILED', 'CANCELLED']:
                    self.send_alert(f"Job {job_status['status']}", 
                                  "success" if job_status['status'] == 'COMPLETED' else "error")
                    break
                
                # Check training progress
                progress = self.check_training_progress()
                
                if 'current_step' in progress:
                    self.logger.info(f"Step {progress['current_step']}, Loss: {progress.get('current_loss', 'N/A')}")
                
                # Health checks
                runtime_hours = (time.time() - self.start_time) / 3600
                if runtime_hours > 24 and progress.get('current_step', 0) < 100:
                    self.send_alert("Training appears stalled", "warning")
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                self.logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(60)  # Wait before retry

# Usage
if __name__ == "__main__":
    import sys
    job_id = sys.argv[1] if len(sys.argv) > 1 else os.getenv('SLURM_JOB_ID')
    webhook = os.getenv('SLACK_WEBHOOK_URL')  # Optional alerts
    
    monitor = ClusterTrainingMonitor(job_id, webhook)
    monitor.run_monitoring()
```

(tutorial-multi-node-step8-troubleshooting)=
## Step 8: Troubleshoot Multi-Node Issues

**Common Enterprise Deployment Issues:**

```bash
# Network connectivity debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET,INIT

# Container access debugging
srun --nodes=2 --ntasks-per-node=1 \
  --container-image=nvcr.io/nvidia/nemo:dev \
  --container-mounts=/shared:/shared \
  hostname

# Shared storage verification
srun --nodes=4 --ntasks-per-node=1 \
  ls -la /shared/models /shared/datasets

# InfiniBand verification (if available)
srun --nodes=2 --ntasks-per-node=1 ibstatus
```

**Performance Optimization for Production:**

```yaml
# Production-optimized configuration
distributed:
  _target_: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager
  dp_size: none

# Network optimizations
nccl_params:
  nccl_ib_disable: 0              # Enable InfiniBand
  nccl_net_gdr_level: 3           # GPU Direct RDMA
  nccl_algo: Tree,Ring            # Communication algorithms

# Memory optimizations
step_scheduler:
  grad_acc_steps: 32              # Large accumulation for multi-node
  ckpt_every_steps: 500           # Frequent checkpointing

# I/O optimizations
dataloader:
  num_workers: 16                 # More workers for shared storage
  persistent_workers: true        # Keep workers alive
  prefetch_factor: 8              # Larger prefetch for network storage
```

(tutorial-multi-node-success-metrics)=
## Enterprise Deployment Success Metrics

**Production Readiness Checklist:**

- [ ] **Multi-node scaling verified**: Training launches successfully across cluster nodes
- [ ] **Container deployment working**: NVIDIA containers run with proper mounts
- [ ] **Shared storage configured**: All nodes can access training data and checkpoints
- [ ] **Monitoring implemented**: Job status and training progress tracked
- [ ] **Failure recovery tested**: Checkpoint restoration and job restart procedures
- [ ] **Resource accounting**: GPU usage tracked for cost allocation

**Real-World Performance Expectations:**

| Configuration | Training Time | Cost Efficiency | Resource Utilization |
|---------------|---------------|-----------------|---------------------|
| **Single Node (8xA100)** | 100% baseline | 1.0x | 95% GPU utilization |
| **2 Nodes (16xA100)** | 52% of baseline | 0.96x efficiency | 90% GPU utilization |
| **4 Nodes (32xA100)** | 28% of baseline | 0.89x efficiency | 85% GPU utilization |
| **8 Nodes (64xA100)** | 16% of baseline | 0.78x efficiency | 75% GPU utilization |

**Enterprise Value Delivered:**

- **Faster Training**: 4-6x speedup on large models with multi-node scaling
- **Resource Efficiency**: 75-90% GPU utilization across cluster
- **Operational Excellence**: Automated job management and monitoring
- **Cost Predictability**: Clear resource usage tracking and billing
- **Compliance Ready**: Audit trails and job logging for enterprise requirements

(tutorial-multi-node-deployment-templates)=
## Production Deployment Templates

**Enterprise Training Config Template:**

```yaml
# enterprise_template.yaml - Production training configuration
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: "{{ model_name }}"
  torch_dtype: torch.bfloat16

distributed:
  _target_: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager
  dp_size: none

slurm:
  job_name: "{{ job_name }}"
  nodes: "{{ node_count }}"
  ntasks_per_node: 8
  time: "{{ time_limit }}"
  account: "{{ billing_account }}"
  partition: "{{ gpu_partition }}"
  container_image: "nvcr.io/nvidia/nemo:{{ version }}"
  hf_home: "/shared/models/.cache"
  extra_mounts:
    - "/shared/datasets:/data"
    - "/shared/results:/results"

checkpoint:
  enabled: true
  checkpoint_dir: "/shared/results/{{ job_name }}/checkpoints"
  model_save_format: safetensors
  async_save: true

# Template usage:
# sed 's/{{ model_name }}/meta-llama\/Llama-3.2-7B/g' enterprise_template.yaml | \
# sed 's/{{ job_name }}/llm_training_$(date +%Y%m%d)/g' | \
# sed 's/{{ node_count }}/4/g' > production_config.yaml
```

(tutorial-multi-node-next-steps)=
## Next Steps for Enterprise Teams

**Immediate Implementation:**
1. **Validate cluster setup** - Test single-node then multi-node training
2. **Configure shared storage** - Ensure all nodes can access training data
3. **Set up monitoring** - Implement job tracking and alert systems
4. **Train team members** - Share access patterns and troubleshooting guides

**Deep Dive into Enterprise Features:**

- **[Advanced Slurm Configuration](../../guides/launcher/slurm.md)** - Production cluster optimization
- **[Checkpointing Strategies](../../guides/checkpointing.md)** - Multi-node state management
- **[Enterprise Use Cases](../use-cases/cluster-administrators.md)** - Real-world deployment patterns

**Apply in Production:**

- **[Distributed Training Example](../examples/distributed-training.md)** - Complete enterprise workflow
- **[Memory-Efficient Training](../examples/memory-efficient-training.md)** - Multi-node PEFT deployment
- **[DevOps Use Cases](../use-cases/devops-professionals.md)** - Infrastructure automation patterns

**API References:**

- **[Launcher Components](../../api-docs/launcher/launcher.md)** - Job submission and management APIs
- **[Distributed Training](../../api-docs/distributed/distributed.md)** - Multi-node coordination
- **[Cluster Management](../../api-docs/utils/utils.md)** - Resource management utilities

**Related Documentation:**

- **[Installation for Clusters](../../get-started/installation.md#cluster-installation)** - Environment setup
- **[Troubleshooting Multi-Node](../../references/troubleshooting-reference.md#multi-node-issues)** - Common issues and solutions

---

**Navigation:**
- ← [Previous: Memory-Efficient Training](parameter-efficient-fine-tuning.md)
- ↑ [Back to Tutorials Overview](index.md)
- → **Complete!** You've mastered enterprise AI training infrastructure

**Congratulations!** You've mastered production-scale distributed training. Your enterprise can now deploy large-scale AI training with confidence, efficiency, and enterprise-grade reliability.
