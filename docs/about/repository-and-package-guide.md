---
description: "Complete guide to NeMo Automodel repository and package structure, from high-level organization to detailed module hierarchy and development patterns."
tags: ["repository", "package", "structure", "development", "components", "architecture"]
categories: ["architecture", "development"]
---

(repository-and-package-guide)=
# Repository & Package Guide

Comprehensive guide to the NeMo Automodel codebase organization, from repository structure to package internals and development workflows.

## Guide Overview

This guide provides complete coverage of NeMo Automodel's organization:

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`repo;1.5em;sd-mr-1` Repository Structure
:link: #repository-structure
:link-type: ref

Top-level repository organization, directories, and development workflow
+++
{bdg-primary}`Start Here`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Package Structure  
:link: #package-structure
:link-type: ref

Deep dive into `nemo_automodel/` module hierarchy and component details
+++
{bdg-info}`Technical Details`
:::

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` Getting Started Paths
:link: #getting-started-paths
:link-type: ref

Recommended exploration paths for different user types
+++
{bdg-success}`Navigation`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Development Patterns
:link: #development-patterns
:link-type: ref

Best practices, design principles, and contribution guidelines
+++
{bdg-warning}`Contributors`
:::

::::

## Repository Structure

The NeMo Automodel repository is organized into several top-level directories, each serving a specific purpose in the development, testing, and deployment workflow.

### Complete Repository Overview

```
Automodel/
├── nemo_automodel/          # Core source code (main package)
│   ├── components/          # Modular training components  
│   ├── recipes/             # End-to-end training workflows
│   ├── _cli/                # Command-line interface
│   └── shared/              # Cross-cutting utilities
├── examples/                # Working YAML configs & sample scripts
│   ├── llm/                 # Language model examples
│   └── vlm/                 # Vision-language model examples  
├── tests/                   # Comprehensive test suites
│   ├── unit_tests/          # Component-level tests
│   └── functional_tests/    # End-to-end integration tests
├── docs/                    # Documentation source & extensions
│   ├── _extensions/         # Custom Sphinx extensions
│   └── guides/              # User guides and tutorials
├── docker/                  # Container definitions & scripts
├── scripts/                 # Development utilities & tools
├── LICENSE                  # Apache 2.0 license
├── CONTRIBUTING.md          # Contribution guidelines
├── pyproject.toml          # Package configuration
└── README.md               # Project overview
```

### Directory Descriptions

```{list-table}
:header-rows: 1
:widths: 20 30 50

* - Directory
  - Purpose
  - Key Contents
* - `nemo_automodel/`
  - **Core Package**
  - Main source code, components, recipes, CLI
* - `examples/`
  - **Working Examples**
  - Production-ready YAML configs, sample scripts
* - `tests/`
  - **Quality Assurance**
  - Unit tests, functional tests, CI/CD validation
* - `docs/`
  - **Documentation**
  - User guides, API docs, tutorials, extensions
* - `docker/`
  - **Containerization**
  - Docker images for development and deployment
* - `scripts/`
  - **Development Tools**
  - Utilities for RAG, indexing, preview generation
```

### Key Files

- **`pyproject.toml`** - Package dependencies, build configuration, tool settings
- **`CONTRIBUTING.md`** - Guidelines for contributors, development setup
- **`LICENSE`** - Apache 2.0 open source license
- **`CHANGELOG.md`** - Version history and release notes

### Examples Directory

The `examples/` directory contains production-ready YAML configurations and sample scripts that demonstrate how to use NeMo Automodel for various training scenarios.

#### Structure

```
examples/
├── llm/
│   ├── finetune.py                     # LLM fine-tuning script
│   ├── llama_3_2_1b_hellaswag_fp8.yaml # Llama with FP8 quantization
│   ├── llama_3_2_1b_hellaswag_nvfsdp.yaml # Llama with nvFSDP
│   └── *.yaml                          # Additional model configurations
└── vlm/
    ├── finetune.py                     # VLM fine-tuning script
    ├── gemma_3_vl_4b_cord_v2_nvfsdp.yaml # Gemma VL with nvFSDP
    ├── gemma_3_vl_4b_cord_v2_peft.yaml   # Gemma VL with PEFT
    └── *.yaml                          # Additional VLM configurations
```

#### Usage

Each YAML configuration can be run directly with the provided scripts:

```bash
# LLM fine-tuning example
python examples/llm/finetune.py --config examples/llm/llama_3_2_1b_hellaswag_nvfsdp.yaml

# VLM fine-tuning example  
python examples/vlm/finetune.py --config examples/vlm/gemma_3_vl_4b_cord_v2_peft.yaml
```

### Testing Structure

The `tests/` directory contains comprehensive test suites that validate all aspects of NeMo Automodel, from individual components to full end-to-end workflows.

#### Test Organization

```
tests/
├── unit_tests/                    # Component-level testing
│   ├── _cli/                      # CLI module tests
│   ├── _peft/                     # PEFT implementation tests
│   ├── _transformers/             # Model wrapper tests
│   ├── checkpoint/                # Checkpointing logic tests
│   ├── config/                    # Configuration system tests
│   ├── datasets/                  # Dataset loader tests
│   ├── distributed/               # Distributed training tests
│   ├── launcher/                  # Job launcher tests
│   ├── loggers/                   # Logging utility tests
│   ├── loss/                      # Loss function tests
│   ├── optim/                     # Optimizer tests
│   ├── quantization/              # FP8 quantization tests
│   ├── recipes/                   # Recipe logic tests
│   ├── shared/                    # Shared utility tests
│   ├── training/                  # Training utility tests
│   └── utils/                     # General utility tests
└── functional_tests/              # End-to-end integration testing
    ├── checkpoint/                # Checkpoint format validation
    ├── hf_consolidated_fsdp/      # HuggingFace + FSDP integration
    ├── hf_dcp/                    # Distributed checkpoint testing
    ├── hf_peft/                   # PEFT integration testing
    ├── hf_transformer/            # Transformer integration tests
    ├── hf_transformer_finetune/   # Fine-tuning workflow tests
    ├── hf_transformer_llm/        # LLM-specific tests
    └── hf_transformer_vlm/        # VLM-specific tests
```

#### Running Tests

```bash
# Run all unit tests
python -m pytest tests/unit_tests/

# Run functional tests (requires GPU)
python -m pytest tests/functional_tests/

# Run specific component tests
python -m pytest tests/unit_tests/datasets/
```

## Package Structure

NeMo Automodel is organized as a modular Python package with clear separation of concerns and well-defined interfaces between components. The package follows NVIDIA's established patterns for ML frameworks while optimizing for the specific needs of fine-tuning and training workflows.

### Complete Package Directory Structure

```text
nemo_automodel/
├── __init__.py                    # Package entry point with lazy loading
├── package_info.py               # Version and metadata
├── _cli/                         # Command-line interface
│   └── app.py                    # Main CLI application
├── components/                   # Core modular components
│   ├── _transformers/            # Hugging Face model integration
│   │   ├── auto_model.py         # NeMoAutoModelForCausalLM, NeMoAutoModelForImageTextToText
│   │   └── utils.py              # Transformer utilities
│   ├── _peft/                    # Parameter-efficient fine-tuning
│   │   ├── lora.py               # LoRA implementation
│   │   ├── lora_kernel.py        # Optimized LoRA kernels
│   │   └── module_matcher.py     # Module matching utilities
│   ├── datasets/                 # Data loading and processing
│   │   ├── utils.py              # Common dataset utilities
│   │   ├── llm/                  # Language model datasets
│   │   │   ├── column_mapped_text_instruction_dataset.py
│   │   │   ├── hellaswag.py      # HellaSwag evaluation dataset
│   │   │   ├── squad.py          # SQuAD question answering
│   │   │   ├── packed_sequence.py # Packed sequence optimization
│   │   │   ├── mock.py           # Mock datasets for testing
│   │   │   └── mock_packed.py    # Mock packed datasets
│   │   └── vlm/                  # Vision-language datasets
│   │       ├── datasets.py       # VLM dataset implementations
│   │       ├── collate_fns.py    # Specialized collation functions
│   │       └── utils.py          # VLM-specific utilities
│   ├── distributed/              # Multi-GPU and distributed training
│   │   ├── ddp.py                # Distributed Data Parallel
│   │   ├── fsdp2.py              # Fully Sharded Data Parallel v2
│   │   ├── nvfsdp.py             # NVIDIA's optimized FSDP
│   │   ├── optimized_tp_plans.py # Tensor parallelism strategies
│   │   ├── parallelizer.py       # Parallelization orchestration
│   │   ├── tensor_utils.py       # Tensor manipulation utilities
│   │   ├── grad_utils.py         # Gradient handling
│   │   ├── cp_utils.py           # Context parallelism utilities
│   │   └── init_utils.py         # Distributed initialization
│   ├── checkpoint/               # Advanced checkpointing
│   │   ├── checkpointing.py      # Main checkpointing logic
│   │   ├── stateful_wrappers.py  # Stateful component wrappers
│   │   ├── _torch_backports.py   # PyTorch compatibility
│   │   └── _backports/           # HuggingFace integration backports
│   │       ├── filesystem.py     # Filesystem abstractions
│   │       ├── hf_storage.py     # HuggingFace storage integration
│   │       ├── hf_utils.py       # HuggingFace utilities
│   │       ├── default_planner.py # Default checkpoint planning
│   │       ├── planner_helpers.py # Checkpoint planning utilities
│   │       ├── consolidate_hf_safetensors.py # SafeTensors consolidation
│   │       ├── _fsspec_filesystem.py # FSSpec filesystem support
│   │       └── _version.py       # Version compatibility
│   ├── loss/                     # Optimized loss functions
│   │   ├── chunked_ce.py         # Chunked cross-entropy
│   │   ├── linear_ce.py          # Linear cross-entropy
│   │   ├── masked_ce.py          # Masked cross-entropy
│   │   ├── te_parallel_ce.py     # Transformer Engine parallel CE
│   │   └── triton/               # Custom Triton kernels
│   │       └── te_cross_entropy.py # TE cross-entropy kernel
│   ├── training/                 # Training utilities and management
│   │   ├── rng.py                # Random number generation
│   │   ├── step_scheduler.py     # Training step scheduling
│   │   ├── timers.py             # Performance timing
│   │   └── utils.py              # Training utilities
│   ├── config/                   # Configuration management
│   │   ├── loader.py             # YAML configuration loading
│   │   └── _arg_parser.py        # Command-line argument parsing
│   ├── launcher/                 # Job launcher and cluster integration
│   │   └── slurm/                # SLURM cluster support
│   │       ├── config.py         # SLURM configuration
│   │       ├── template.py       # Job template generation
│   │       └── utils.py          # SLURM utilities
│   ├── loggers/                  # Logging and monitoring
│   │   ├── wandb_utils.py        # Weights & Biases integration
│   │   └── log_utils.py          # General logging utilities
│   ├── optim/                    # Optimization algorithms
│   │   └── scheduler.py          # Learning rate schedulers
│   ├── quantization/             # Model quantization
│   │   └── fp8.py                # FP8 quantization support
│   └── utils/                    # Component utilities
│       ├── dist_utils.py         # Distributed utilities
│       ├── model_utils.py        # Model manipulation
│       ├── sig_utils.py          # Signature utilities
│       └── yaml_utils.py         # YAML processing
├── recipes/                      # End-to-end training workflows
│   ├── base_recipe.py            # Base recipe class
│   ├── llm/                      # Language model recipes
│   │   └── finetune.py           # LLM fine-tuning recipe
│   └── vlm/                      # Vision-language model recipes
│       └── finetune.py           # VLM fine-tuning recipe
└── shared/                       # Cross-component shared utilities
    ├── import_utils.py           # Safe import utilities
    └── utils.py                  # Common utility functions
```

### Module Details

#### Core Components (`components/`)

##### **Model Integration (`_transformers/`)**
- **Purpose**: Bridge between Hugging Face models and NeMo training infrastructure
- **Key Classes**: `NeMoAutoModelForCausalLM`, `NeMoAutoModelForImageTextToText`
- **Features**: Drop-in replacements with optimized kernels and distributed support

##### **Parameter-Efficient Fine-Tuning (`_peft/`)**
- **Purpose**: LoRA and other PEFT implementations with optimized kernels
- **Key Components**: LoRA layers, kernel optimizations, module matching
- **Integration**: Works seamlessly with any supported model architecture

##### **Data Pipeline (`datasets/`)**
- **LLM Datasets**: Instruction datasets, evaluation benchmarks, packed sequences
- **VLM Datasets**: Vision-language datasets with specialized preprocessing
- **Features**: Optimized collation, memory-efficient loading, flexible transforms

##### **Distributed Training (`distributed/`)**
- **Strategies**: DDP, FSDP2, nvFSDP, tensor parallelism
- **Features**: Automatic strategy selection, gradient optimization, communication efficiency
- **Scaling**: Single GPU to multi-node clusters

#### Training Recipes (`recipes/`)

##### **Base Recipe Architecture**
```python
class BaseRecipe:
    def __init__(self, model, dataset, strategy, config):
        # Common initialization
    
    def setup(self):
        # Environment and component setup
    
    def train(self):
        # Training loop implementation
    
    def evaluate(self):
        # Evaluation logic
    
    def checkpoint(self):
        # State management
```

##### **LLM Recipes (`llm/`)**
- **Fine-tuning**: Full and parameter-efficient fine-tuning
- **Optimization**: Automatic mixed precision, gradient accumulation
- **Features**: Model-specific optimizations, memory management

##### **VLM Recipes (`vlm/`)**
- **Vision Language Training**: Multi-modal fine-tuning workflows
- **Data Handling**: Image-text pair processing, efficient batching
- **Memory Optimization**: Large model support with gradient checkpointing

#### Shared Infrastructure

##### **Import Management (`shared/import_utils.py`)**
- **Safe Imports**: Graceful handling of optional dependencies
- **Feature Detection**: Runtime capability discovery
- **Fallbacks**: Alternative implementations when dependencies unavailable

##### **Configuration System (`components/config/`)**
- **YAML-driven**: Human-readable configuration files
- **Validation**: Schema validation and error reporting
- **Templating**: Reusable configuration patterns

##### **CLI Interface (`_cli/`)**
- **Job Launching**: Simple command-line interface for training
- **Environment Detection**: Automatic cluster and GPU detection
- **Configuration**: CLI argument to YAML configuration mapping

## Getting Started Paths

Choose your exploration path based on your role and goals:

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` New Users
:class-header: sd-bg-success sd-text-white

**Recommended Path:**
1. 📁 Start with `examples/` - Review working YAML configurations
2. 🧪 Run a simple training example
3. 📖 Read {doc}`/get-started/quick-start`
4. 🔧 Modify configurations for your use case

+++
{bdg-success}`Beginner Friendly`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Contributors & Developers
:class-header: sd-bg-primary sd-text-white

**Recommended Path:**
1. 🧩 Explore `nemo_automodel/components/` - Understand building blocks
2. 🍳 Study `nemo_automodel/recipes/` - See component orchestration
3. 🧪 Run `tests/` - Understand expected behavior
4. 📋 Read `CONTRIBUTING.md` - Development guidelines

+++
{bdg-primary}`Technical`
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Advanced Users
:class-header: sd-bg-info sd-text-white

**Recommended Path:**
1. 📦 Study package architecture - Component design patterns
2. 🔧 Examine distributed training strategies
3. ⚡ Understand optimization techniques
4. 🏗️ Create custom components or recipes

+++
{bdg-info}`Advanced`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Infrastructure Teams
:class-header: sd-bg-warning sd-text-white

**Recommended Path:**
1. 🐳 Review `docker/` - Container setup
2. 🎯 Understand launcher and SLURM integration
3. ⚖️ Study distributed training strategies
4. 📊 Examine logging and monitoring setup

+++
{bdg-warning}`Operations`
:::

::::

### Core Package Structure
The Automodel source code is available under the [`nemo_automodel`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/nemo_automodel) directory. It is organized into three main directories:
- [`components/`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/nemo_automodel/components) - Self-contained modules
- [`recipes/`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/nemo_automodel/recipes) - End-to-end training workflows
- [`_cli/`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/nemo_automodel/_cli) - Command-line interface

#### Components Directory
The `components/` directory contains isolated modules used in training loops. Each component is designed to be dependency-light and reusable without cross-module imports.

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

#### Key Component Features
- Each component can be used independently in other projects
- Each component has its own dependencies, without cross-module imports
- Unit tests are colocated with the component they cover

#### Recipes Directory
Recipes define **end-to-end workflows** (data → training → eval) for a variety of tasks, combining components into usable pipelines.

```
$ tree -L 2 nemo_automodel/recipes/
├── llm
│   └── finetune.py   - Finetune recipe for LLMs (SFT, PEFT).
└── vlm
    └── finetune.py   - Finetune recipe for VLMs (SFT, PEFT).
```

For configuration examples and running instructions, see {ref}`get-started-quick-start` and the LLM SFT guide.

#### CLI Directory
The `automodel` CLI simplifies job execution across environments. See the Quick Start guide for basic examples and the SLURM launcher guide for cluster usage.

## Development Patterns

### Component Design Principles

#### **Independence**
- Each component has minimal external dependencies
- Clear interfaces with well-defined contracts
- Self-contained testing and validation

#### **Composability**
- Components work together through standard interfaces
- Mix-and-match different implementations
- Recipe-level orchestration of component interactions

#### **Extensibility**
- New components follow established patterns
- Inheritance from base classes
- Plugin-style architecture for custom implementations

### Import Architecture

#### **Lazy Loading**
```python
def __getattr__(name: str):
    if name in __all__:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

#### **Top-level Promotion**
Key classes are promoted to package level for convenient access via the main package namespace.

#### **Safe Dependencies**
Optional dependencies handled gracefully:
```python
try:
    from optional_dependency import SomeClass
    HAS_OPTIONAL = True
except ImportError:
    HAS_OPTIONAL = False
    SomeClass = None
```

### Configuration Patterns

#### **YAML-First Design**
- All training parameters in YAML files
- Runtime override capabilities
- Environment variable substitution
- Validation and error reporting

#### **Hierarchical Structure**
```yaml
model:
  name: "meta-llama/Llama-3.2-1B"
  config_overrides:
    attention_dropout: 0.1

data:
  dataset_type: "instruction"
  batch_size: 32

training:
  optimizer: "adamw"
  learning_rate: 3e-4
  num_epochs: 3

distributed:
  strategy: "fsdp2"
  tensor_parallel_size: 1
```

### Best Practices for Contributors

#### Adding New Components

1. **Follow Naming Conventions**: Use descriptive, consistent names
2. **Implement Base Interfaces**: Inherit from established base classes
3. **Add Comprehensive Tests**: Unit tests and integration tests
4. **Document APIs**: Clear docstrings with examples
5. **Update Configuration**: Add config options if needed

#### Testing Patterns

- **Unit Tests**: Test individual component functionality
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete training workflows
- **Performance Tests**: Validate scaling and memory usage

#### Documentation Requirements

- **API Documentation**: Auto-generated from docstrings
- **Usage Examples**: Practical code examples
- **Configuration Reference**: All available options
- **Migration Guides**: For breaking changes

## Development Workflow

Understanding the repository structure helps with effective development:

1. **Start with Examples**: Use `examples/` to understand expected usage patterns
2. **Modify Components**: Make changes in `nemo_automodel/components/` for new features
3. **Update Recipes**: Modify `nemo_automodel/recipes/` for workflow changes
4. **Add Tests**: Create tests in `tests/` that mirror your changes
5. **Update Documentation**: Modify `docs/` to reflect new features
6. **Container Testing**: Use `docker/` for reproducible testing environments

## Summary

NeMo Automodel's structure is designed for developer productivity, component reusability, and production scalability. The modular architecture enables rapid prototyping while the well-defined interfaces ensure reliable integration patterns. This structure supports the framework's goal of providing immediate access to new models with enterprise-grade training capabilities.

### Quick Reference

- **Repository Overview**: Start with `examples/` and `README.md`
- **Package Internals**: Explore `nemo_automodel/components/` and `recipes/`
- **Testing**: Run tests in `tests/` to understand behavior
- **Documentation**: Comprehensive guides in `docs/`
- **Development**: Follow patterns in `CONTRIBUTING.md`
- **Deployment**: Use containers in `docker/` for consistent environments
