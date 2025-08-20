---
description: "Practical guide to navigating the NeMo Automodel repository and package structure, with detailed module breakdown and development workflows."
tags: ["repository", "package", "navigation", "development", "modules", "workflow"]
categories: ["development", "reference"]
---

(repository-and-package-guide)=
# Repository & Package Guide

Practical guide to navigating and understanding the NeMo Automodel codebase organization, from finding files to understanding module relationships and contributing effectively.

## Guide Overview

This guide provides complete coverage of NeMo Automodel's organization:

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`repo;1.5em;sd-mr-1` Repository Structure
:link: repository-structure
:link-type: ref

Top-level repository organization, directories, and development workflow
+++
{bdg-primary}`Start Here`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Package Structure  
:link: package-structure
:link-type: ref

Deep dive into `nemo_automodel/` module hierarchy and component details
+++
{bdg-info}`Technical Details`
:::

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` Getting Started Paths
:link: getting-started-paths
:link-type: ref

Recommended exploration paths for different user types
+++
{bdg-success}`Navigation`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Development Patterns
:link: development-patterns
:link-type: ref

Best practices, design principles, and contribution guidelines
+++
{bdg-warning}`Contributors`
:::

::::

(repository-structure)=
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

(package-structure)=
## Package Structure

This section provides a detailed breakdown of the `nemo_automodel/` package structure to help you navigate and understand the codebase effectively.

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
│   │   ├── te_parallel_ce.py     # Tensor-parallel cross-entropy loss
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

### Detailed Module Breakdown

#### Model Integration Layer (`_transformers/`)

**File Structure:**
```
_transformers/
├── __init__.py
├── auto_model.py    # Main model wrapper classes
└── utils.py         # Transformer utilities
```

**Key Classes and Functions:**
- `NeMoAutoModelForCausalLM` - Language model wrapper
- `NeMoAutoModelForImageTextToText` - Vision-language model wrapper  
- `_BaseNeMoAutoModelClass` - Base wrapper functionality
- `patch_model_with_kernel_optimizations()` - Applies automatic optimizations

**Usage Patterns:**
```python
# Direct model loading
from nemo_automodel import NeMoAutoModelForCausalLM
model = NeMoAutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# With optimization configuration
model = NeMoAutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    use_liger_kernel=True,
    attn_implementation="flash_attention_2"
)
```

#### Parameter-Efficient Fine-Tuning (`_peft/`)

**File Structure:**
```
_peft/
├── __init__.py
├── lora.py           # LoRA implementation with config
├── lora_kernel.py    # Optimized LoRA kernels
└── module_matcher.py # Pattern matching for module targeting
```

**Key Components:**
- `PeftConfig` - Configuration for PEFT methods
- `LoRALayer` - Low-rank adaptation layer implementation
- `ModuleMatcher` - Flexible module selection patterns
- Triton-optimized kernels for performance

**Configuration Examples:**
```yaml
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  match_all_linear: true
  dim: 32
  alpha: 64
  use_triton: true
```

#### Data Processing Pipeline (`datasets/`)

**File Structure:**
```
datasets/
├── __init__.py
├── utils.py          # Common dataset utilities
├── llm/             # Language model datasets
│   ├── column_mapped_text_instruction_dataset.py
│   ├── hellaswag.py
│   ├── squad.py
│   ├── packed_sequence.py
│   ├── mock.py       # Testing datasets
│   └── mock_packed.py
└── vlm/             # Vision-language datasets
    ├── datasets.py   # VLM dataset implementations
    ├── collate_fns.py # Specialized collation
    └── utils.py      # VLM utilities
```

**Dataset Categories:**
- **Instruction Datasets**: Text-based instruction following
- **Evaluation Datasets**: Benchmarks like HellaSwag, SQuAD
- **Packed Sequences**: Memory-efficient sequence packing
- **VLM Datasets**: Multi-modal image-text datasets
- **Mock Datasets**: Testing and development datasets

#### Distributed Training Infrastructure (`distributed/`)

**File Structure:**
```
distributed/
├── __init__.py
├── ddp.py                    # Distributed Data Parallel
├── fsdp2.py                  # Fully Sharded Data Parallel v2
├── nvfsdp.py                 # NVIDIA optimized FSDP
├── optimized_tp_plans.py     # Tensor parallelism plans
├── parallelizer.py           # Parallelization orchestration
├── tensor_utils.py           # Tensor operations
├── grad_utils.py             # Gradient handling
├── cp_utils.py               # Context parallelism
└── init_utils.py             # Distributed initialization
```

**Strategy Selection:**
```yaml
# DDP for smaller models
distributed:
  _target_: nemo_automodel.components.distributed.ddp.DDPManager

# FSDP2 for larger models  
distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  
# nvFSDP for production scaling
distributed:
  _target_: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager
```

#### Advanced Checkpointing (`checkpoint/`)

**File Structure:**
```
checkpoint/
├── __init__.py
├── checkpointing.py           # Main checkpoint logic
├── stateful_wrappers.py       # Component state wrappers
├── _torch_backports.py        # PyTorch compatibility
└── _backports/               # HuggingFace integrations
    ├── filesystem.py
    ├── hf_storage.py
    ├── consolidate_hf_safetensors.py
    └── ...
```

**Checkpoint Formats:**
- **HuggingFace Format**: Compatible with HF Hub
- **Distributed Checkpoint (DCP)**: Sharded for large models
- **Consolidated Format**: Single-file deployment format
- **SafeTensors**: Secure tensor serialization

#### Training Orchestration (`recipes/`)

**File Structure:**
```
recipes/
├── __init__.py
├── base_recipe.py    # Abstract base recipe
├── llm/
│   └── finetune.py   # LLM fine-tuning workflow
└── vlm/
    └── finetune.py   # VLM fine-tuning workflow
```

**Recipe Inheritance Pattern:**
```python
class BaseRecipe:
    def setup(self): pass
    def train(self): pass  
    def evaluate(self): pass

class LLMFinetuneRecipe(BaseRecipe):
    # LLM-specific implementation
    
class VLMFinetuneRecipe(BaseRecipe):
    # VLM-specific implementation
```

#### Supporting Infrastructure

**Configuration Management (`config/`):**
```
config/
├── __init__.py
├── loader.py         # YAML configuration loading
└── _arg_parser.py    # CLI argument parsing
```

**Job Launching (`launcher/`):**
```
launcher/
└── slurm/
    ├── config.py     # SLURM configuration
    ├── template.py   # Job template generation
    └── utils.py      # SLURM utilities
```

**Logging and Monitoring (`loggers/`):**
```
loggers/
├── __init__.py
├── wandb_utils.py    # Weights & Biases integration
└── log_utils.py      # General logging utilities
```

(getting-started-paths)=
## Get Started Paths

Choose your exploration path based on your role and goals:

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` New Users
:class-header: sd-bg-success sd-text-white

**Recommended Path:**
1. 📁 Start with `examples/` - Review working YAML configurations
2. 🧪 Run a simple training example
3. 📖 Read {doc}`../get-started/quick-start`
4. 🔧 Modify configurations for your use case

+++
{bdg-success}`Beginner Friendly`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Contributors & Developers
:class-header: sd-bg-primary sd-text-white

**Recommended Path:**
1. 🧩 Explore `nemo_automodel/components/` - Understand building blocks
2. 🍳 Study `nemo_automodel/recipes/` - Refer to component orchestration
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

### Navigation Quick Reference

**Finding Files by Function:**

```{list-table}
:header-rows: 1
:widths: 30 70

* - Looking for...
  - Check these locations
* - Model loading/wrapping
  - `nemo_automodel/components/_transformers/`
* - PEFT configurations
  - `nemo_automodel/components/_peft/`  
* - Dataset implementations
  - `nemo_automodel/components/datasets/llm/` or `datasets/vlm/`
* - Distributed training setup
  - `nemo_automodel/components/distributed/`
* - Training workflows
  - `nemo_automodel/recipes/llm/` or `recipes/vlm/`
* - CLI commands
  - `nemo_automodel/_cli/app.py`
* - Configuration loading
  - `nemo_automodel/components/config/`
* - Checkpointing logic
  - `nemo_automodel/components/checkpoint/`
* - Working examples
  - `examples/llm/` or `examples/vlm/`
* - Unit tests
  - `tests/unit_tests/[component_name]/`
* - Integration tests
  - `tests/functional_tests/`
```

### File Naming Conventions

**Components follow consistent patterns:**
- `__init__.py` - Package initialization with imports
- `*.py` - Implementation files with descriptive names
- `utils.py` - Utility functions for the component
- `config.py` - Configuration classes (where applicable)

**Common file patterns:**
- `*_dataset.py` - Dataset implementations
- `*_utils.py` - Utility functions
- `test_*.py` - Unit tests (in corresponding test directories)

### Module Import Patterns

**Top-level imports (promoted to package namespace):**
```python
from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel import NeMoAutoModelForImageTextToText
```

**Component-level imports:**
```python
from nemo_automodel.components._peft.lora import PeftConfig
from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager
```

**Recipe imports:**
```python
from nemo_automodel.recipes.llm.finetune import LLMFinetuneRecipe
```

(development-patterns)=
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

Understanding the repository structure enables effective development:

### **Step-by-Step Development Process**

1. **📁 Explore Examples First**
   ```bash
   # Start with working examples to understand patterns
   ls examples/llm/  # Review LLM configurations
   ls examples/vlm/  # Review VLM configurations
   ```

2. **🔍 Identify Target Component** 
   ```bash
   # Find the right component for your change
   find nemo_automodel/components/ -name "*.py" | grep -i [your_feature]
   ```

3. **🧪 Run Existing Tests**
   ```bash
   # Understand current behavior
   python -m pytest tests/unit_tests/[component_name]/
   ```

4. **✏️ Make Changes**
   ```bash
   # Edit component files
   vim nemo_automodel/components/[component]/[file].py
   ```

5. **🔧 Update Tests**
   ```bash
   # Add/modify tests to cover your changes  
   vim tests/unit_tests/[component]/test_[file].py
   ```

6. **✅ Validate Changes**
   ```bash
   # Run tests to ensure everything works
   python -m pytest tests/unit_tests/[component]/
   python -m pytest tests/functional_tests/ -k [relevant_test]
   ```

### **Common Development Patterns**

**Adding a New Dataset:**
1. Implement in `nemo_automodel/components/datasets/llm/` or `datasets/vlm/`
2. Add unit tests in `tests/unit_tests/datasets/`
3. Create example configuration in `examples/`
4. Update documentation

**Adding a New Distributed Strategy:**
1. Implement in `nemo_automodel/components/distributed/`
2. Add integration tests in `tests/functional_tests/`
3. Update recipes to support new strategy
4. Add performance benchmarks

**Extending PEFT Methods:**
1. Implement in `nemo_automodel/components/_peft/`
2. Add kernel optimizations if needed
3. Create comprehensive tests
4. Add example configurations

### **Testing Strategy**

**Unit Tests** - Test individual components:
```bash
# Test specific component
python -m pytest tests/unit_tests/datasets/ -v

# Test with coverage
python -m pytest tests/unit_tests/datasets/ --cov=nemo_automodel.components.datasets
```

**Functional Tests** - Test end-to-end workflows:
```bash
# Test LLM training workflow
python -m pytest tests/functional_tests/hf_transformer_llm/

# Test distributed training
python -m pytest tests/functional_tests/hf_consolidated_fsdp/
```

**Integration Testing** - Test component interactions:
```bash
# Test PEFT + distributed training
python -m pytest tests/functional_tests/hf_peft/
```

### **Documentation Updates**

**API Documentation:**
- Update docstrings in source files
- Auto-generated via Sphinx

**User Guides:**
- Add examples to `docs/guides/`
- Update tutorials in `docs/tutorials/`

**Configuration Reference:**
- Update YAML schema documentation
- Add configuration examples

### Debugging and Troubleshooting

**Common Investigation Paths:**

**Import Issues:**
```bash
# Check import dependencies
python -c "from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager"
```

**Configuration Problems:**
```bash
# Validate YAML configuration
python -c "from nemo_automodel.components.config.loader import load_config; load_config('your_config.yaml')"
```

**Component Behavior:**
```bash
# Test individual components
python -m pytest tests/unit_tests/[component]/ -v -s
```

**Performance Issues:**
```bash
# Profile training components
python -m cProfile -o profile_output.prof your_training_script.py
```