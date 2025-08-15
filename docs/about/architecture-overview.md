---
description: "Deep dive into the technical architecture and design principles of NeMo Automodel"
tags: ["architecture", "technical", "design", "components"]
categories: ["concepts"]
---

(about-architecture-overview)=
# Architecture Overview

```{raw} html
<style>
.clickable-diagram {
    cursor: pointer;
    transition: transform 0.2s ease-in-out;
    border: 2px solid #4A90E2;
    border-radius: 8px;
    padding: 10px;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
}

.clickable-diagram:hover {
    transform: scale(1.02);
    box-shadow: 0 4px 12px rgba(74, 144, 226, 0.3);
}

.clickable-diagram:active {
    transform: scale(0.98);
}

/* Modal styles for expanded view */
.diagram-modal {
    display: none;
    position: fixed;
    z-index: 1000;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0,0,0,0.8);
    backdrop-filter: blur(5px);
}

.diagram-modal-content {
    position: relative;
    margin: 5% auto;
    padding: 20px;
    width: 90%;
    max-width: 1200px;
    background: white;
    border-radius: 12px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.diagram-modal-close {
    position: absolute;
    top: 10px;
    right: 20px;
    font-size: 28px;
    font-weight: bold;
    color: #666;
    cursor: pointer;
    transition: color 0.2s;
}

.diagram-modal-close:hover {
    color: #333;
}

.diagram-modal img {
    width: 100%;
    height: auto;
    border-radius: 8px;
}
</style>

<script>
document.addEventListener('DOMContentLoaded', function() {
    // Add click handlers to all diagrams
    const diagrams = document.querySelectorAll('.clickable-diagram');
    
    diagrams.forEach(function(diagram) {
        diagram.addEventListener('click', function() {
            // Create modal
            const modal = document.createElement('div');
            modal.className = 'diagram-modal';
            
            // Get the title from data attribute or use default
            const title = this.getAttribute('data-title') || 'Architecture Diagram';
            
            modal.innerHTML = `
                <div class="diagram-modal-content">
                    <span class="diagram-modal-close">&times;</span>
                    <h3>${title}</h3>
                    <div style="max-height: 80vh; overflow-y: auto;">${this.innerHTML}</div>
                </div>
            `;

            document.body.appendChild(modal);
            modal.style.display = 'block';
            document.body.style.overflow = 'hidden';

            // Close modal functionality
            const closeBtn = modal.querySelector('.diagram-modal-close');
            closeBtn.addEventListener('click', function() {
                modal.remove();
                document.body.style.overflow = 'auto';
            });

            // Close on outside click
            modal.addEventListener('click', function(e) {
                if (e.target === modal) {
                    modal.remove();
                    document.body.style.overflow = 'auto';
                }
            });
        });
    });
    
    // Close on Escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            const modals = document.querySelectorAll('.diagram-modal');
            modals.forEach(function(modal) {
                modal.remove();
                document.body.style.overflow = 'auto';
            });
        }
    });
});
</script>
```

NeMo Automodel is built on a modular, component-based architecture that bridges Hugging Face models with NVIDIA's high-performance training ecosystem. This overview explores the technical design, key components, and architectural decisions that enable Day-0 support for new models while maintaining enterprise-grade scalability and performance.

## Core Design Principles

The architecture is guided by several key principles:

- **Modularity**: Self-contained components with minimal cross-dependencies
- **Compatibility**: Drop-in replacement for Hugging Face AutoModel classes
- **Performance**: Optimized kernels and distributed training strategies
- **Flexibility**: YAML-driven configuration with extensible component system
- **Scalability**: Support for single GPU to multi-node clusters

## High-Level Architecture

::::{grid} 1 1 1 1
:gutter: 2

:::{grid-item-card} {octicon}`stack;2em;sd-mr-1` Layered Architecture
:class-header: sd-bg-primary sd-text-white

```{raw} html
<div class="clickable-diagram" data-title="NeMo Automodel Layered Architecture">
```

```{mermaid}
graph TB
    A[Recipes Layer] --> B[Components Layer]
    B --> C[Transformers Layer]
    C --> D[Hugging Face Models]
    
    A1[LLM Recipe] --> A
    A2[VLM Recipe] --> A
    
    B1[Datasets] --> B
    B2[Distributed] --> B
    B3[Training] --> B
    B4[PEFT] --> B
    B5[Loss] --> B
    B6[Optim] --> B
    
    C1[NeMoAutoModelForCausalLM] --> C
    C2[NeMoAutoModelForImageTextToText] --> C
    
    D1[AutoModelForCausalLM] --> D
    D2[AutoModelForImageTextToText] --> D
```

```{raw} html
</div>
```
:::

::::

## Entry Points and Model Wrappers

### Auto Model Classes

NeMo Automodel provides drop-in replacements for Hugging Face's AutoModel classes:

- **`NeMoAutoModelForCausalLM`**: For language models (LLMs)
- **`NeMoAutoModelForImageTextToText`**: For vision language models (VLMs)

These classes inherit from `_BaseNeMoAutoModelClass`, which extends Hugging Face's `_BaseAutoModelClass` to add NeMo-specific optimizations:

```python
class NeMoAutoModelForCausalLM(_BaseNeMoAutoModelClass, AutoModelForCausalLM):
    """Drop-in replacement with custom kernels and optimizations"""
```

### Key Features

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Kernel Optimizations
- **Liger Kernels**: Automatic patching for faster attention
- **SDPA Integration**: Scaled Dot-Product Attention backends
- **Flash Attention 2**: Optimized attention implementation
- **FP8 Quantization**: Hardware-accelerated precision
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration Options
- **`use_liger_kernel`**: Enable/disable Liger optimizations
- **`use_sdpa_patching`**: SDPA attention patching
- **`fp8_config`**: FP8 quantization settings
- **`attn_implementation`**: Attention backend selection
:::

::::

## Component Architecture

The components directory contains **11 core modules**, each designed as self-contained, reusable units:

### Core Components

```{list-table}
:header-rows: 1
:widths: 20 80

* - Component
  - Purpose
* - `_transformers/`
  - Optimized Hugging Face model wrappers and utilities
* - `_peft/`
  - Parameter-Efficient Fine-Tuning (LoRA, DoRA implementations)
* - `datasets/`
  - LLM and VLM dataset loaders with preprocessing utilities
* - `distributed/`
  - Distributed training strategies (DDP, FSDP2, nvFSDP)
* - `checkpoint/`
  - Advanced checkpoint save/load with format conversion
* - `config/`
  - YAML configuration loading and CLI argument parsing
* - `loss/`
  - Optimized loss functions (chunked cross-entropy, linear CE)
* - `optim/`
  - Optimizers and learning rate schedulers
* - `training/`
  - Training utilities, RNG management, step scheduling
* - `launcher/`
  - Job launchers for Slurm and Kubernetes environments
* - `utils/`
  - Lightweight utilities for timing, profiling, and filesystem operations
```

### Component Design Philosophy

Each component follows strict design guidelines:

- **Dependency-light**: Minimal external dependencies
- **No cross-imports**: Components don't import from each other
- **Unit tested**: Colocated tests with components
- **Configurable**: YAML-driven instantiation via `_target_` pattern

## Recipes: End-to-End Workflows

Recipes orchestrate components into complete training pipelines:

### Recipe Architecture

```{raw} html
<div class="clickable-diagram" data-title="Recipe Architecture">
```

```{mermaid}
graph LR
    A[Configuration] --> B[Recipe Setup]
    B --> C[Component Instantiation]
    C --> D[Training Loop]
    D --> E[Checkpointing]
    E --> F[Evaluation]
    
    subgraph "Components Used"
        G[Model + Optimizer]
        H[Datasets]
        I[Distributed]
        J[Loss Function]
        K[LR Scheduler]
    end
    
    C --> G
    C --> H
    C --> I
    C --> J
    C --> K
```

```{raw} html
</div>
```

### Available Recipes

- **`llm/finetune.py`**: LLM fine-tuning (SFT, PEFT)
- **`vlm/finetune.py`**: VLM fine-tuning (SFT, PEFT)

### Recipe Workflow

1. **Setup Phase**: Initialize distributed environment, logging, wandb
2. **Component Building**: Instantiate model, optimizer, datasets, schedulers
3. **Training Loop**: Execute forward/backward passes with gradient accumulation
4. **Checkpointing**: Save/load model state and optimizer state
5. **Evaluation**: Run validation steps during training

## Distributed Training Strategies

NeMo Automodel supports three distributed training approaches:

### Strategy Comparison

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} {octicon}`cpu;1.5em;sd-mr-1` DDP
:class-header: sd-bg-success sd-text-white

**Data Parallel**
- Simplest multi-GPU setup
- Each GPU holds full model copy
- Gradient synchronization via AllReduce
- Best for smaller models
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` FSDP2
:class-header: sd-bg-info sd-text-white

**Fully Sharded Data Parallel**
- Parameters sharded across GPUs
- Optional tensor parallelism
- Mixed precision support
- CPU offloading capabilities
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` nvFSDP
:class-header: sd-bg-warning sd-text-white

**NVIDIA Optimized FSDP**
- Hardware-optimized sharding
- Advanced overlap strategies
- FP8 support integration
- Multi-node scaling
:::

::::

### Parallelism Dimensions

```{raw} html
<div class="clickable-diagram" data-title="Parallelism Dimensions">
```

```{mermaid}
graph TB
    A[Model] --> B[Data Parallel - DP]
    A --> C[Tensor Parallel - TP]
    A --> D[Context Parallel - CP]
    
    B --> B1[Shard Parameters]
    B --> B2[Replicate Across Nodes]
    
    C --> C1[Shard Within Layers]
    C --> C2[Attention Heads]
    C --> C3[Feed Forward]
    
    D --> D1[Sequence Sharding]
    D --> D2[Long Context Support]
```

```{raw} html
</div>
```

## Configuration System

### YAML-Driven Architecture

The configuration system uses a powerful `ConfigNode` class that enables:

- **Object Instantiation**: `_target_` pattern for creating objects
- **Nested Configuration**: Hierarchical YAML structure
- **Dynamic Resolution**: Runtime function/class resolution
- **Validation**: Type checking and parameter validation

### Configuration Flow

```{raw} html
<div class="clickable-diagram" data-title="Configuration Flow">
```

```{mermaid}
graph LR
    A[YAML Config] --> B[ConfigNode]
    B --> C[_target_ Resolution]
    C --> D[Object Instantiation]
    D --> E[Component Assembly]
    
    subgraph "Example"
        F["model:\n  _target_: NeMoAutoModelForCausalLM\n  pretrained_model_name_or_path: meta-llama/Llama-2-7b"]
    end
    
    A --> F
```

```{raw} html
</div>
```

## Performance Optimizations

### Attention Optimizations

- **SDPA Backends**: Automatic backend selection (Flash Attention, Math, Memory-efficient)
- **Liger Kernels**: Monkey-patched attention layers for speed improvements
- **Kernel Fallback**: Graceful degradation if optimizations fail

### Precision and Quantization

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Mixed Precision
- **BF16 Training**: Reduced memory, maintained accuracy
- **FP16 Support**: Automatic loss scaling
- **Dynamic Scaling**: Prevents gradient underflow
:::

:::{grid-item-card} {octicon}`circuit-board;1.5em;sd-mr-1` FP8 Quantization
- **Hardware Acceleration**: H100+ GPU optimization
- **Tensorwise/Rowwise**: Different scaling strategies
- **torchAO Integration**: Leverages PyTorch native FP8
:::

::::

## PEFT Integration

### LoRA Architecture

```{raw} html
<div class="clickable-diagram" data-title="LoRA Architecture">
```

```{mermaid}
graph LR
    A[Base Model] --> B[Linear Layers]
    B --> C[LoRA Injection]
    C --> D[Low-Rank Matrices]
    D --> E[Trainable Parameters]
    
    subgraph "LoRA Decomposition"
        F[W = W₀ + BA]
        G[B: d×r matrix]
        H[A: r×k matrix]
    end
    
    D --> F
    F --> G
    F --> H
```

```{raw} html
</div>
```

### PEFT Features

- **LoRA/DoRA**: Low-rank adaptation with optional DoRA improvements
- **Module Targeting**: Flexible pattern matching for layer selection
- **Rank Configuration**: Adjustable rank for memory/performance trade-offs
- **Dropout Support**: Regularization in adapter layers

## Checkpointing System

### Advanced Checkpoint Features

- **Multiple Formats**: HuggingFace, DCP (Distributed Checkpoint), Consolidated
- **Automatic Conversion**: Seamless format transformations
- **PEFT Integration**: Adapter-aware checkpointing
- **Resume Capability**: Robust training resumption

### Checkpoint Flow

```{raw} html
<div class="clickable-diagram" data-title="Checkpoint Flow">
```

```{mermaid}
graph TB
    A[Training State] --> B{Checkpoint Format}
    B --> C[HuggingFace Format]
    B --> D[DCP Format] 
    B --> E[Consolidated Format]
    
    C --> F[Single File]
    D --> G[Distributed Shards]
    E --> H[Deployment Ready]
    
    F --> I[Easy Loading]
    G --> J[Scalable Storage]
    H --> K[Inference Engines]
```

```{raw} html
</div>
```

## Memory and Scale Management

### Memory Optimization Strategies

- **Gradient Checkpointing**: Trade compute for memory
- **CPU Offloading**: Move parameters/optimizer states to CPU
- **Mixed Precision**: Reduce memory footprint
- **Parameter Sharding**: Distribute model across devices

### Scaling Characteristics

```{list-table}
:header-rows: 1
:widths: 25 25 25 25

* - Strategy
  - Memory Efficiency
  - Communication Overhead
  - Complexity
* - DDP
  - Low
  - Low
  - Simple
* - FSDP2
  - High
  - Medium
  - Medium
* - nvFSDP
  - Very High
  - Optimized
  - Advanced
```

## Integration with Broader Ecosystem

### Upstream Compatibility

- **Hugging Face Hub**: Direct model loading
- **Transformers API**: Compatible interfaces
- **Datasets Library**: Native dataset support
- **Tokenizers**: Automatic tokenizer handling

### Downstream Deployment

- **vLLM Integration**: Export for high-performance inference
- **TensorRT-LLM**: Planned optimization support
- **Standard Formats**: HuggingFace-compatible outputs

## Extensibility and Future Development

### Extension Points

- **Custom Components**: Add new training components
- **Model Support**: Extend to new architectures
- **Optimization Backends**: Integrate new kernel libraries
- **Distributed Strategies**: Add new parallelism approaches

### Roadmap Considerations

- **Pipeline Parallelism**: 4D parallelism support
- **Video Models**: Extension beyond text and vision
- **Additional Backends**: More inference engine integrations
- **Advanced Optimizations**: Continued performance improvements

## Package Structure Deep Dive

### Core Module Organization

NeMo Automodel follows a hierarchical modular architecture with clear separation of concerns:

```text
nemo_automodel/
├── components/           # Modular training components
│   ├── _transformers/    # HF model integration layer
│   ├── _peft/           # Parameter-efficient fine-tuning
│   ├── datasets/        # Data loading and processing
│   │   ├── llm/         # Language model datasets
│   │   └── vlm/         # Vision-language datasets
│   ├── distributed/     # Multi-GPU training strategies
│   ├── checkpoint/      # Advanced state management
│   ├── loss/            # Optimized loss functions
│   ├── training/        # Training utilities & scheduling
│   ├── launcher/        # Job launching & cluster integration
│   ├── loggers/         # Monitoring & experiment tracking
│   ├── optim/          # Optimization algorithms
│   ├── quantization/    # Model quantization support
│   └── utils/          # Component-specific utilities
├── recipes/             # End-to-end training workflows
│   ├── llm/            # Language model recipes
│   └── vlm/            # Vision-language model recipes
├── shared/             # Cross-component utilities
└── _cli/               # Command-line interface
```

### Component Interaction Patterns

#### **Dependency Flow**
1. **Shared Utilities** → Foundation layer used by all components
2. **Core Components** → Independent, composable building blocks  
3. **Training Recipes** → Orchestrate components into complete workflows
4. **CLI Interface** → User-facing entry point for launching workflows

#### **Import Architecture**
- **Top-level imports**: Key classes promoted to package namespace (`NeMoAutoModelForCausalLM`, `NeMoAutoModelForImageTextToText`)
- **Lazy loading**: Submodules imported on-demand for fast startup
- **Safe imports**: Graceful handling of optional dependencies with fallbacks
- **Clear interfaces**: Well-defined APIs between components

### Key Design Principles

#### **Modularity**
Each component is self-contained with minimal cross-dependencies:
- **Transformers**: Independent model wrappers with drop-in HF compatibility
- **PEFT**: Standalone LoRA implementations with optimized kernels
- **Distributed**: Pluggable parallelization strategies (DDP, FSDP2, nvFSDP)
- **Datasets**: Reusable data processing pipelines for LLM and VLM

#### **Composability** 
Components combine seamlessly for custom workflows through well-defined interfaces and standardized configuration patterns.

#### **Extensibility**
New components integrate easily by:
- Inheriting from base classes (`BaseRecipe`, component interfaces)
- Following established patterns (configuration, checkpointing, logging)
- Using shared utilities (import management, YAML processing)
- Maintaining API consistency across the ecosystem

## Summary

NeMo Automodel's architecture successfully bridges the gap between Hugging Face's model ecosystem and NVIDIA's high-performance training infrastructure. The modular design enables rapid experimentation while the optimized components ensure production-ready performance. This architectural foundation supports the platform's core mission: providing Day-0 access to new models with enterprise-grade training capabilities.