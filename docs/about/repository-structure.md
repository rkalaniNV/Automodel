# NeMo Automodel Repository

This introductory guide presents the structure of the NeMo Automodel repository, provides a brief overview of its parts, introduces concepts such as components and recipes, and explains how everything fits together.

## Repository Structure
The Automodel source code is available under the [`nemo_automodel`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/nemo_automodel) directory. It is organized into three directories:
- [`components/`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/nemo_automodel/components)  - Self-contained modules
- [`recipes/`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/nemo_automodel/recipes) - End-to-end training workflows
- [`cli/`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/nemo_automodel/_cli) - launch fine-tuning jobs.


### Components Directory
The `components/` directory contains isolated modules used in training loops.
Each component is designed to be dependency-light and reusable without cross-module imports.

#### Directory Structure
The following directory listing shows all components along with explanations of their contents.
```
$ tree -L 1 nemo_automodel/components/

├── _peft/          - Implementations of PEFT methods, such as LoRA.
├── _transformers/  - Optimized model implementations for Hugging Face models.
├── checkpoint/     - Checkpoint save and load-related logic.
├── config/         - Utils to load YAML files and CLI-parsing helpers.
├── datasets/       - LLM and VLM datasets and utils (collate functions, preprocessing).
├── distributed/    - Distributed processing primitives (DDP, FSDP2, nvFSDP).
├── launcher/       - Job launcher for interactive and batch (Slurm, K8s) processing.
├── loggers/        - Metric/event logging for Weights & Biases and other tools
├── loss/           - Loss functions (such as cross-entropy and linear cross-entropy, etc.).
├── optim/          - Optimizers and LR schedulers, including fused or second-order variants.
├── training/       - Training and fine-tuning utils.
└── utils/          - Small, dependency-free helpers (seed, profiler, timing, fs).
```

#### Key Features
- Each component can be used independently in other projects.
- Each component has its own dependencies, without cross-module imports.
- Unit tests are colocated with the component they cover.

### Recipes Directory
Recipes define **end-to-end workflows** (data → training → eval) for a variety of tasks, such as,
training, fine-tuning, knowledge distillation, and combining components into usable pipelines.

#### Available Recipes
The following directory listing shows all components along with explanations of their contents.
```
$ tree -L 2 nemo_automodel/recipes/
├── llm
│   └── finetune.py   - Finetune recipe for LLMs (SFT, PEFT).
└── vlm
    └── finetune.py   - Finetune recipe for VLMs (SFT, PEFT).
```

#### Run Recipes

For how to run recipes with `torchrun` or the `automodel` CLI, see the {ref}`get-started-quick-start` and the LLM SFT guide.

<!-- For an in-depth explanation of the LLM recipe please also see the [LLM recipe deep-dive guide](docs/llm_recipe_deep_dive.md). -->

#### Configure a Recipe
For YAML configuration examples, refer to the LLM SFT guide and the examples in the repository.

### CLI Directory
The `automodel` CLI simplifies job execution across environments. See Quick Start for a basic example and the Slurm launcher guide for cluster usage.
<!-- The [Automodel CLI guide](docs/automodel_cli.md) provides an in-depth explanation of the automodel util. -->
