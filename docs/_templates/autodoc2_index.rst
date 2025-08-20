API Reference
=============

{{ product_name }}'s API reference provides comprehensive technical documentation for all modules, classes, and functions. Use these references to understand the technical foundation of NeMo Automodel and integrate it with your model training workflows.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`package;1.5em;sd-mr-1` Model Components
      :link: _transformers/_transformers
      :link-type: doc
      :class-card: sd-border-0

      **Model Wrappers & Interfaces**

      Core model classes with optimizations and Hugging Face integration for efficient training.

      :bdg-secondary:`NeMoAutoModelForCausalLM` :bdg-secondary:`optimizations` :bdg-secondary:`wrappers`

   .. grid-item-card:: :octicon:`database;1.5em;sd-mr-1` Dataset Loaders
      :link: datasets/datasets
      :link-type: doc
      :class-card: sd-border-0

      **LLM & VLM Data Processing**

      Dataset loaders for language models and vision language tasks with custom collation functions.

      :bdg-secondary:`squad` :bdg-secondary:`medpix` :bdg-secondary:`collate-fns` :bdg-secondary:`instruction`

   .. grid-item-card:: :octicon:`server;1.5em;sd-mr-1` Distributed Training
      :link: distributed/distributed
      :link-type: doc
      :class-card: sd-border-0

      **Multi-GPU & Cluster Strategies**

      FSDP2, DDP, and nvFSDP managers for scalable distributed training across multiple nodes.

      :bdg-secondary:`FSDP2` :bdg-secondary:`tensor-parallel` :bdg-secondary:`context-parallel` :bdg-secondary:`nvFSDP`

   .. grid-item-card:: :octicon:`zap;1.5em;sd-mr-1` PEFT & Fine-tuning
      :link: _peft/_peft
      :link-type: doc
      :class-card: sd-border-0

      **Parameter-Efficient Training**

      LoRA adapters and efficient fine-tuning configurations with custom kernel implementations.

      :bdg-secondary:`LoRA` :bdg-secondary:`triton-kernels` :bdg-secondary:`adapters`

   .. grid-item-card:: :octicon:`graph;1.5em;sd-mr-1` Loss Functions
      :link: loss/loss
      :link-type: doc
      :class-card: sd-border-0

      **Optimized Loss Computation**

      Memory-efficient cross-entropy and specialized loss functions with Triton acceleration.

      :bdg-secondary:`chunked-ce` :bdg-secondary:`masked-ce` :bdg-secondary:`triton` :bdg-secondary:`parallel-ce`

   .. grid-item-card:: :octicon:`gear;1.5em;sd-mr-1` Training Utilities
      :link: training/training
      :link-type: doc
      :class-card: sd-border-0

      **Scheduling & Management**

      Step schedulers, RNG management, and training utilities for robust training workflows.

      :bdg-secondary:`step-scheduler` :bdg-secondary:`rng` :bdg-secondary:`timers` :bdg-secondary:`utils`

   .. grid-item-card:: :octicon:`rocket;1.5em;sd-mr-1` Optimizers
      :link: optim/optim
      :link-type: doc
      :class-card: sd-border-0

      **Optimization Algorithms**

      Advanced optimizers and learning rate schedulers for efficient model convergence.

      :bdg-secondary:`optimizers` :bdg-secondary:`schedulers` :bdg-secondary:`lr-scheduling` :bdg-secondary:`convergence`

   .. grid-item-card:: :octicon:`database;1.5em;sd-mr-1` Checkpointing
      :link: checkpoint/checkpoint
      :link-type: doc
      :class-card: sd-border-0

      **Training State Management**

      Safetensors format support and PyTorch DCP checkpointing with Hugging Face compatibility.

      :bdg-secondary:`safetensors` :bdg-secondary:`DCP` :bdg-secondary:`huggingface` :bdg-secondary:`resume`

   .. grid-item-card:: :octicon:`tools;1.5em;sd-mr-1` Configuration
      :link: config/config
      :link-type: doc
      :class-card: sd-border-0

      **Configuration Management**

      YAML configuration loading and argument parsing for flexible training setup.

      :bdg-secondary:`yaml` :bdg-secondary:`argparse` :bdg-secondary:`config` :bdg-secondary:`setup`

   .. grid-item-card:: :octicon:`play;1.5em;sd-mr-1` Job Launcher
      :link: launcher/launcher
      :link-type: doc
      :class-card: sd-border-0

      **Multi-Environment Deployment**

      SLURM cluster integration and distributed training coordination across environments.

      :bdg-secondary:`slurm` :bdg-secondary:`cluster` :bdg-secondary:`deployment` :bdg-secondary:`coordination`

   .. grid-item-card:: :octicon:`chart-line;1.5em;sd-mr-1` Logging
      :link: loggers/loggers
      :link-type: doc
      :class-card: sd-border-0

      **Training Monitoring**

      Comprehensive logging utilities with WandB integration for experiment tracking.

      :bdg-secondary:`wandb` :bdg-secondary:`logging` :bdg-secondary:`monitoring` :bdg-secondary:`tracking`

   .. grid-item-card:: :octicon:`cpu;1.5em;sd-mr-1` Quantization
      :link: quantization/quantization
      :link-type: doc
      :class-card: sd-border-0

      **Model Quantization**

      FP8 quantization and precision optimization for enhanced performance and memory efficiency.

      :bdg-secondary:`FP8` :bdg-secondary:`quantization` :bdg-secondary:`precision` :bdg-secondary:`efficiency`

   .. grid-item-card:: :octicon:`tools;1.5em;sd-mr-1` Utilities
      :link: utils/utils
      :link-type: doc
      :class-card: sd-border-0

      **Helper Functions**

      Distribution utilities, model helpers, and common tools for development workflows.

      :bdg-secondary:`dist-utils` :bdg-secondary:`model-utils` :bdg-secondary:`helpers` :bdg-secondary:`tools`

.. toctree::
   :maxdepth: 1
   :caption: API Modules
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