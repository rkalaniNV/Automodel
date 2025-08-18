# API Reference

NeMo Automodel's API reference provides comprehensive technical documentation for all components, classes, and functions used in training workflows.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Model Components
:link: _transformers
:link-type: doc
:class-card: sd-border-0

**Model Wrappers & Interfaces**

Core model classes with optimizations and Hugging Face integration.

{bdg-primary}`NeMoAutoModelForCausalLM` {bdg-secondary}`optimizations`
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Dataset Loaders
:link: datasets
:link-type: doc
:class-card: sd-border-0

**LLM & VLM Data Processing**

Dataset loaders for language models and vision language tasks.

{bdg-secondary}`squad` {bdg-secondary}`medpix` {bdg-secondary}`collate-fns`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Distributed Training
:link: distributed
:link-type: doc
:class-card: sd-border-0

**Multi-GPU & Cluster Strategies**

FSDP2, DDP, and nvFSDP managers for scalable training.

{bdg-info}`FSDP2` {bdg-info}`tensor-parallel` {bdg-info}`context-parallel`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` PEFT & Fine-tuning
:link: _peft
:link-type: doc
:class-card: sd-border-0

**Parameter-Efficient Training**

LoRA adapters and efficient fine-tuning configurations.

{bdg-success}`LoRA` {bdg-success}`DoRA` {bdg-secondary}`triton-kernels`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Loss Functions
:link: loss
:link-type: doc
:class-card: sd-border-0

**Optimized Loss Computation**

Memory-efficient cross-entropy and specialized loss functions.

{bdg-warning}`chunked-ce` {bdg-warning}`masked-ce` {bdg-secondary}`triton`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Training Utilities
:link: training
:link-type: doc
:class-card: sd-border-0

**Scheduling & Management**

Step schedulers, RNG management, and training utilities.

{bdg-secondary}`step-scheduler` {bdg-secondary}`rng` {bdg-secondary}`timers`
:::

::::

```{toctree}
:maxdepth: 1
:caption: Component APIs
:hidden:

_transformers
datasets
distributed
_peft
loss
training
optim
checkpoint
config
launcher
loggers
quantization
utils
```
