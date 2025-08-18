# ML Engineers Use Cases

Advanced use cases for ML engineers focused on model performance, efficiency, and multi-modal capabilities using NeMo Automodel.

:::{note}
**Target Audience**: Intermediate ML engineers  
**Focus**: Performance optimization, multi-modal models, production readiness
:::

## Overview

As an ML engineer, you're concerned with model performance, efficiency, and integrating various data types. NeMo Automodel's multi-modal support and optimization features enable sophisticated AI solutions for production environments.

---

## Use Case 1: Multi-modal Content Analysis for E-commerce

**Business Context**: E-commerce platform needs to automatically generate product descriptions and classify products based on both product images and existing text descriptions.

### Problem Statement
- Manual product categorization takes 10-15 minutes per item
- Product descriptions are inconsistent across vendors
- Need to process both images and text for accurate classification
- Must handle 10,000+ new products daily

### NeMo Automodel Solution

**Step 1: Multi-modal Dataset Preparation**
```python
# Product data structure
{
  "product_id": "prod_12345",
  "image_path": "/images/prod_12345.jpg",
  "description": "Bluetooth wireless headphones with noise canceling",
  "category": "Electronics/Audio/Headphones",
  "attributes": ["wireless", "noise-canceling", "bluetooth"]
}
```

**Step 2: Vision-Language Model Configuration**
```yaml
# product_classification_vlm.yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: microsoft/kosmos-2-patch14-224
  torch_dtype: bfloat16

data:
  _target_: nemo_automodel.datasets.vlm.ProductClassificationDataset
  train_file: "products_train.jsonl"
  val_file: "products_val.jsonl"
  batch_size: 4
  image_size: 224
  max_seq_length: 512
  
# Multi-modal fusion strategy
fusion_strategy: "cross_attention"
vision_encoder_frozen: false  # Allow vision adaptation

optimizer:
  _target_: torch.optim.AdamW
  lr: 1e-5  # Lower LR for VLM fine-tuning
  weight_decay: 0.01

scheduler:
  _target_: nemo_automodel.optim.scheduler.CosineAnnealingScheduler
  max_steps: 2000
  warmup_steps: 200
```

**Step 3: Advanced Training with Mixed Precision**
```bash
# Optimized training for performance
automodel finetune vlm -c product_classification_vlm.yaml \
  --precision bf16 \
  --gradient_checkpointing \
  --compile_model
```

### Advanced Features Implementation

**Caption Generation + Classification**
```python
# Dual-task training for richer understanding
class ProductAnalysisModel:
    def __init__(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)
    
    def analyze_product(self, image, existing_description=""):
        # Generate enhanced description
        caption_prompt = f"Describe this product in detail: {existing_description}"
        inputs = self.processor(text=caption_prompt, images=image, return_tensors="pt")
        caption_outputs = self.model.generate(**inputs, max_new_tokens=150)
        enhanced_description = self.processor.decode(caption_outputs[0])
        
        # Classify product
        class_prompt = f"Category: {enhanced_description}"
        class_inputs = self.processor(text=class_prompt, images=image, return_tensors="pt")
        class_outputs = self.model.generate(**class_inputs, max_new_tokens=50)
        category = self.processor.decode(class_outputs[0])
        
        return enhanced_description, category
```

### Learning Outcomes
- **Multi-modal Architecture**: Understanding vision-language model design
- **Cross-modal Fusion**: Learning how visual and textual features combine
- **Performance Optimization**: Mixed precision and gradient checkpointing
- **Production Integration**: Building robust inference pipelines

### Results
- **Classification Accuracy**: 94% (vs 78% text-only, 85% image-only)
- **Processing Speed**: 2.3 seconds per product (vs 15 minutes manual)
- **Description Quality**: 89% vendor approval rating
- **Business Impact**: $2.4M annual savings in content creation costs

---

## Use Case 2: Model Optimization and Pruning for Mobile Deployment

**Business Context**: Healthcare company needs to deploy a medical image analysis model on mobile devices for remote diagnosis in areas with poor connectivity.

### Problem Statement
- Full model (2.8GB) too large for mobile deployment
- Need <100MB model with minimal accuracy loss
- Must maintain 95%+ accuracy for safety-critical application
- Target inference time <200ms on mobile CPU

### NeMo Automodel Solution

**Step 1: Baseline Model Training**
```yaml
# medical_imaging_full.yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForImageClassification.from_pretrained
  pretrained_model_name_or_path: microsoft/swin-large-patch4-window7-224
  num_labels: 12  # Different medical conditions

data:
  _target_: nemo_automodel.datasets.vision.MedicalImageDataset
  train_dir: "medical_images/train"
  val_dir: "medical_images/val"
  batch_size: 16
  image_size: 224

trainer:
  max_epochs: 20
  precision: bf16
```

**Step 2: Knowledge Distillation**
```yaml
# medical_imaging_distilled.yaml
teacher_model:
  model_path: "medical_full_model"
  
student_model:
  _target_: nemo_automodel.NeMoAutoModelForImageClassification.from_pretrained
  pretrained_model_name_or_path: microsoft/swin-tiny-patch4-window7-224
  num_labels: 12

distillation:
  temperature: 4.0
  alpha: 0.7  # Weight for distillation loss
  
optimizer:
  _target_: torch.optim.AdamW
  lr: 5e-5
```

**Step 3: Quantization and Pruning**
```python
# Post-training optimization pipeline
import torch.quantization as quant
from nemo_automodel.optimization import PruningConfig

# Load distilled model
model = AutoModelForImageClassification.from_pretrained("medical_distilled")

# Apply structured pruning
pruning_config = PruningConfig(
    pruning_ratio=0.3,
    pruning_type="structured",
    importance_metric="magnitude"
)
pruned_model = apply_pruning(model, pruning_config)

# Apply quantization
quantized_model = torch.quantization.quantize_dynamic(
    pruned_model,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)

# Validate accuracy retention
accuracy = evaluate_model(quantized_model, test_dataset)
print(f"Optimized model accuracy: {accuracy:.2%}")
```

**Step 4: Mobile Deployment Optimization**
```python
# Convert to mobile-optimized format
import torch.jit as jit

# TorchScript compilation
traced_model = jit.trace(quantized_model, example_input)
traced_model.save("medical_model_mobile.pt")

# ONNX export for cross-platform deployment
torch.onnx.export(
    quantized_model,
    example_input,
    "medical_model.onnx",
    export_params=True,
    opset_version=11,
    input_names=['image'],
    output_names=['diagnosis'],
    dynamic_axes={'image': {0: 'batch_size'}}
)
```

### Learning Outcomes
- **Model Compression**: Knowledge distillation and pruning techniques
- **Quantization**: Understanding precision trade-offs
- **Mobile Optimization**: TorchScript and ONNX deployment
- **Performance Profiling**: Measuring inference speed and memory usage

### Results
- **Model Size**: 2.8GB → 89MB (97% reduction)
- **Accuracy**: 96.4% → 95.8% (0.6% drop)
- **Inference Time**: 1.2s → 180ms (6.7x speedup)
- **Deployment**: Successfully runs on iPhone 12+ and Android devices

---

## Use Case 3: Advanced Memory Optimization for Vision-Language Models

**Business Context**: Research team wants to train large Vision-Language Models but faces severe memory constraints that prevent training on their existing 8x RTX 4090 cluster.

### Problem Statement  
- VLMs require significantly more memory than LLMs due to vision components
- Standard VLM training needs 40GB+ per GPU (requires A100-80GB)
- Research team has 8x RTX 4090 (24GB each) and wants to utilize full cluster
- Need to train competitive VLM models without upgrading hardware

### NeMo AutoModel Advanced VLM Solution

**Step 1: VLM Memory Analysis and Optimization Strategy**
```python
# vlm_memory_analyzer.py
def analyze_vlm_memory_requirements():
    """Analyze memory requirements for VLM training"""
    
    # Standard VLM memory breakdown
    vision_encoder_gb = 8.5    # Vision transformer + image processing
    language_model_gb = 14.0   # 7B language model  
    cross_attention_gb = 4.2   # Multi-modal fusion layers
    optimizer_gb = 26.7        # AdamW states for all parameters
    activations_gb = 12.8      # Forward pass activations
    
    total_standard = sum([vision_encoder_gb, language_model_gb, 
                         cross_attention_gb, optimizer_gb, activations_gb])
    
    print(f"Standard VLM Training: {total_standard:.1f}GB")
    
    # NeMo AutoModel PEFT + Optimization
    frozen_vision_gb = 8.5     # Frozen vision encoder (no gradients)
    language_peft_gb = 1.2     # LoRA adapters only
    cross_attention_peft_gb = 0.3  # LoRA on fusion layers
    peft_optimizer_gb = 1.5    # Optimizer states for adapters only
    optimized_activations_gb = 6.4  # Flash attention + checkpointing
    
    total_optimized = sum([frozen_vision_gb, language_peft_gb,
                          cross_attention_peft_gb, peft_optimizer_gb,
                          optimized_activations_gb])
    
    print(f"NeMo AutoModel PEFT: {total_optimized:.1f}GB")
    print(f"Memory reduction: {((total_standard - total_optimized) / total_standard) * 100:.1f}%")

analyze_vlm_memory_requirements()
# Output: Standard: 66.2GB | NeMo PEFT: 17.9GB | Reduction: 73%
```

**Step 2: Memory-Optimized VLM Configuration**
```yaml
# memory_efficient_vlm_training.yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained
  pretrained_model_name_or_path: microsoft/git-large-vqav2
  torch_dtype: torch.bfloat16
  attn_implementation: flash_attention_2

# Selective PEFT for VLM components
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  match_all_linear: false        # Selective targeting for VLMs
  include_modules:
    # Adapt language components only, freeze vision
    - "*.language_model.*.self_attn.*"
    - "*.language_model.*.mlp.*"
    - "*.multi_modal_projector.*"  # Cross-modal fusion
  dim: 16                        # Conservative rank for memory
  alpha: 32
  use_triton: true

# Advanced distributed strategy for VLM
distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: none                  # Use all 8 GPUs

# Memory-efficient VLM dataset
dataset:
  _target_: nemo_automodel.components.datasets.vlm.datasets.make_vqa_dataset
  dataset_name: "your_vqa_dataset"
  split: train
  image_size: 224                # Smaller images for memory
  max_text_length: 256          # Shorter sequences
  
# Conservative training for VLM
step_scheduler:
  grad_acc_steps: 16             # Large accumulation for small batches
  max_steps: 2000
  val_every_steps: 200

dataloader:
  batch_size: 1                  # Very conservative per GPU
  num_workers: 4
  pin_memory: true

# Memory optimizations
training_optimizations:
  gradient_checkpointing: true   # Trade compute for memory
  freeze_vision_encoder: true    # Freeze vision components
```

**Step 3: Advanced Monitoring**
```python
# Real-time performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.gpu_stats = []
        self.throughput_history = []
        
    def log_step_metrics(self, step, loss, gpu_memory, throughput):
        metrics = {
            'step': step,
            'loss': loss,
            'gpu_memory_gb': gpu_memory / 1e9,
            'tokens_per_second': throughput,
            'gpu_utilization': get_gpu_utilization()
        }
        
        # Log to Weights & Biases or TensorBoard
        wandb.log(metrics)
        
        # Alert if performance degrades
        if throughput < self.throughput_history[-10:].mean() * 0.8:
            self.alert_performance_drop()

monitor = PerformanceMonitor()
```

### Advanced Infrastructure Engineering Results
- **Memory Engineering**: 73% reduction enables VLM training on RTX 4090s
- **Cluster Utilization**: Full 8-GPU cluster utilization vs impossible before
- **Scaling Strategy**: Linear scaling across cluster with FSDP2 sharding
- **Infrastructure ROI**: Use existing hardware for models previously requiring A100-80GB

### Results & Infrastructure Transformation
- **VLM Training**: Now possible on 8x RTX 4090 (previously impossible)
- **Memory Usage**: 66GB → 18GB per model (73% reduction)
- **Cluster Efficiency**: 94% GPU utilization across 8 GPUs
- **Training Speed**: 3.2 hours vs 8+ hours on single A100-80GB
- **Infrastructure Savings**: $240,000 saved vs upgrading to A100-80GB cluster

---

## Advanced Techniques

### Multi-GPU Optimization Strategies
```yaml
# Advanced distributed configuration
distributed:
  tensor_parallel_size: 2    # Split model across 2 GPUs
  pipeline_parallel_size: 1  # No pipeline parallelism
  data_parallel_size: 4      # 4-way data parallelism
  
# Communication optimization
nccl_params:
  nccl_p2p_disable: 0
  nccl_tree_threshold: 0
```

### Custom Loss Functions for Multi-modal
```python
class MultiModalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, image_features, text_features):
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(image_features, text_features.T) / self.temperature
        
        # Contrastive loss
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        return (loss_i2t + loss_t2i) / 2
```

## Getting Started

### Prerequisites
- Experience with deep learning frameworks
- Understanding of computer vision and NLP concepts
- Access to multiple GPUs for optimization work
- NeMo Automodel installed ({doc}`../../get-started/installation`)

### Next Steps for Infrastructure-Aware Developers
1. **Memory Breakthrough**: Start with Use Case 1 for 7B model access
2. **Scale Cluster**: Use Case 2 for multi-GPU distributed training
3. **Advanced Models**: Use Case 3 for VLM training on existing hardware
4. **Enterprise Scale**: Progress to {doc}`devops-professionals` for production deployment

### Infrastructure Investment Strategy
- **Week 1**: Validate 7B model training on existing RTX 4090 hardware
- **Week 2**: Scale to multi-GPU distributed training across cluster
- **Week 3**: Advanced VLM training with memory optimizations
- **Week 4**: Calculate ROI and plan next infrastructure investments

### Resources
- {doc}`../../tutorials/parameter-efficient-fine-tuning` - PEFT optimization guide
- {doc}`../../examples/memory-efficient-training` - Working memory optimization examples  
- {doc}`../../tutorials/multi-gpu-training` - Distributed training tutorial

---

**Success Metrics for Infrastructure-Aware AI Developers:**
- **Memory Efficiency**: Train 2-3x larger models on same hardware
- **Cluster Utilization**: 90%+ GPU utilization across infrastructure  
- **Cost Optimization**: 70-85% reduction vs upgrading hardware
- **Model Accessibility**: Enable advanced models for entire team
