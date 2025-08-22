# Tutorials

Practical tutorials for experienced AI developers to achieve immediate performance gains and enterprise-scale deployment with NeMo AutoModel.

## Tutorial Overview

These practical tutorials are designed for experienced AI developers who want to leverage NeMo AutoModel's performance and scaling advantages. Each tutorial delivers immediate, measurable value for production workflows.

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Get 2-3x PyTorch Speedup
:link: first-fine-tuning
:link-type: doc
:link-alt: Performance optimization tutorial

**Applied ML Engineers**: Replace HF Trainer with optimized training for immediate performance gains.
+++
{bdg-primary}`Performance` {bdg-secondary}`20-30 min`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Train 7B Models on 8GB GPUs
:link: parameter-efficient-fine-tuning
:link-type: doc
:link-alt: Memory optimization tutorial

**Infrastructure-Aware Developers**: Memory-efficient training with PEFT + distributed strategies.
+++
{bdg-success}`Efficiency` {bdg-secondary}`45-60 min`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Deploy Multi-Node Training
:link: multi-gpu-training
:link-type: doc
:link-alt: Enterprise deployment tutorial

**Enterprise AI Practitioners**: Production Slurm integration with enterprise monitoring.
+++
{bdg-warning}`Enterprise` {bdg-secondary}`60-90 min`
:::

::::

## Learning Path

Follow our recommended sequence to maximize your infrastructure investments:

### 1. Start Here: {doc}`first-fine-tuning`
**Get 2-3x PyTorch speedup with one config change**. Drop-in replacement for HF Trainer with automatic performance optimizations. Perfect for teams already training with PyTorch/HF.

### 2. Scale Efficiently: {doc}`parameter-efficient-fine-tuning`  
**Train 7B models on 8GB GPUs**. Break through memory limitations with PEFT + distributed training. Essential for teams with GPU resource constraints.

### 3. Enterprise Deployment: {doc}`multi-gpu-training`
**Deploy multi-node training on Slurm clusters today**. Production-ready distributed training with built-in cluster integration. Critical for enterprise AI infrastructure.

## Tutorial Design for Production Teams

Each tutorial delivers immediate business value:

- **Performance Benchmarks**: Measure real improvements vs current workflows
- **Production Configurations**: Enterprise-ready YAML configs and scripts
- **Real Infrastructure**: Works with your existing GPU clusters and environments
- **Business Impact**: Cost savings, efficiency gains, and capability improvements
- **Implementation Guides**: Practical deployment steps for your team

## Immediate ROI for Enterprise Teams

All tutorials focus on production outcomes:

- **Faster Training**: 2-6x speedup vs vanilla PyTorch workflows
- **Memory Efficiency**: Train larger models on existing hardware
- **Cost Reduction**: 60-80% reduction in GPU training time
- **Infrastructure Optimization**: Better utilization of expensive GPU resources
- **Enterprise Integration**: Production deployment with monitoring and compliance

## Summary

These production-focused tutorials deliver competitive advantages for AI teams:

1. **Performance Optimization**: 2-3x speedup over vanilla PyTorch with zero code changes
2. **Memory Breakthrough**: Train 7B+ models on mainstream GPU hardware
3. **Enterprise Scaling**: Multi-node training with production cluster integration

**Your Next Steps:**

- **For Specific Use Cases**: {doc}`../examples/index` - Persona-based examples
- **For Advanced Features**: {doc}`../../guides/index` - Deep technical guides  
- **For Production Reference**: {doc}`../../references/index` - API and CLI specifications

**Key Business Value:**

- **Immediate ROI**: 60-80% reduction in training costs through efficiency gains
- **Infrastructure Optimization**: Maximum utilization of expensive GPU resources
- **Competitive Advantage**: Train larger models faster than teams using vanilla frameworks
- **Enterprise Ready**: Production deployment with monitoring, compliance, and scalability

You're now equipped to transform your AI training infrastructure and deliver measurable business impact through NeMo AutoModel's enterprise-grade optimizations!

```{toctree}
:maxdepth: 2
:hidden:

first-fine-tuning
parameter-efficient-fine-tuning
multi-gpu-training
```
