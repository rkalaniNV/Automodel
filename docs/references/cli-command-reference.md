---
description: "Complete command-line interface reference for NeMo Automodel CLI with all commands, options, and usage patterns"
tags: ["cli", "command-line", "automodel", "finetune", "reference"]
categories: ["reference"]
personas: ["mle-focused", "data-scientist-focused", "enterprise-focused"]
difficulty: "reference"
content_type: "reference"
modality: "universal"
---

(cli-command-reference)=
# CLI Command Reference

Complete reference for the NeMo Automodel command-line interface, covering all commands, options, and usage patterns.

## Overview

The NeMo Automodel CLI provides a unified interface for launching training jobs locally or on distributed systems. It simplifies complex workflows while providing access to advanced configuration options.

### Basic Syntax

```bash
automodel <command> <domain> -c <config_file> [options]
```

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` Commands
:class-header: sd-bg-primary sd-text-white

Available operations:
- `finetune` - Model fine-tuning
- More commands planned
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Domains  
:class-header: sd-bg-info sd-text-white

Model types supported:
- `llm` - Large Language Models
- `vlm` - Vision Language Models
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration
:class-header: sd-bg-success sd-text-white

YAML-driven setup:
- Required `-c/--config` flag
- Override parameters via CLI
:::

::::

## Commands

### `finetune`

Fine-tune pre-trained models with support for various techniques including supervised fine-tuning (SFT) and parameter-efficient fine-tuning (PEFT).

```bash
automodel finetune <domain> -c <config.yaml> [options]
```

**Supported Domains:**
- `llm` - Fine-tune language models (Llama, Gemma, Qwen, etc.)
- `vlm` - Fine-tune vision language models (Gemma 3 VL, etc.)

## Global Options

### Required Arguments

```{list-table}
:header-rows: 1
:widths: 25 75

* - Option
  - Description
* - `<command>`
  - Operation to perform (currently: `finetune`)
* - `<domain>`
  - Model domain (`llm` or `vlm`)
* - `-c, --config PATH`
  - **Required.** Path to YAML configuration file
```

### Optional Arguments

```{list-table}
:header-rows: 1
:widths: 25 15 60

* - Option
  - Type
  - Description
* - `--nproc-per-node INT`
  - integer
  - Number of processes (GPUs) per node. If not specified, uses all available GPUs
* - `--nproc_per_node INT`
  - integer
  - Alternative syntax for `--nproc-per-node`
```

## Usage Examples

### Single GPU Training

Train on a single GPU with minimal configuration:

```bash
# LLM fine-tuning
automodel finetune llm -c examples/llm/llama_3_2_1b_squad.yaml

# VLM fine-tuning  
automodel finetune vlm -c examples/vlm/gemma_3_vl_4b_medpix.yaml
```

### Multi-GPU Training

#### Automatic GPU Detection

Use all available GPUs on the current node:

```bash
automodel finetune llm -c examples/llm/llama_3_2_1b_squad.yaml
```

The CLI automatically detects available GPUs and uses `torchrun` for distributed training.

#### Manual GPU Specification

Specify the exact number of GPUs to use:

```bash
# Use 2 GPUs
automodel finetune llm -c examples/llm/llama_3_2_1b_squad.yaml --nproc-per-node=2

# Use 4 GPUs  
automodel finetune vlm -c examples/vlm/gemma_3_vl_4b_medpix.yaml --nproc-per-node=4
```

### Slurm Cluster Integration

For distributed training on Slurm clusters, add a `slurm` section to your YAML configuration:

**Configuration Example:**
```yaml
# Regular training configuration
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B

# Slurm-specific settings
slurm:
  job_name: llm-finetune
  nodes: 2
  ntasks_per_node: 8
  time: "02:00:00"
  partition: gpu
  account: my_account
```

**Launch Command:**
```bash
automodel finetune llm -c config_with_slurm.yaml
```

The CLI will automatically submit a Slurm batch job instead of running locally.

## Configuration Integration

### YAML Configuration Files

The CLI requires a YAML configuration file that defines:

- **Model specification** (model type, checkpoints, optimizations)
- **Dataset configuration** (data sources, preprocessing)
- **Training parameters** (learning rates, schedulers, epochs)
- **Distributed settings** (parallelism strategies)
- **Optional: Slurm settings** (for cluster deployment)

### Configuration Override

While not directly supported via CLI flags, you can override configuration parameters using the underlying Python scripts with dot-notation:

```bash
# Direct Python execution with overrides
uv run examples/llm/finetune.py \
    --config examples/llm/llama_3_2_1b_squad.yaml \
    --optimizer.lr 2e-5 \
    --step_scheduler.max_steps 1000
```

## Environment Variables

The CLI respects several environment variables:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Variable
  - Description
* - `WANDB_API_KEY`
  - Weights & Biases API key for experiment tracking
* - `HF_TOKEN`  
  - Hugging Face token for accessing gated models
* - `SLURM_*`
  - Various Slurm environment variables (automatically handled)
```

## Error Handling

### Common Issues

**Configuration file not found:**
```
error: argument -c/--config: can't open '<file>': [Errno 2] No such file or directory
```
→ Ensure the YAML configuration file path is correct

**Invalid domain:**
```
error: argument <domain>: invalid choice: 'invalid' (choose from 'llm', 'vlm')
```
→ Use supported domains: `llm` or `vlm`

**Invalid command:**
```
error: argument <command>: invalid choice: 'invalid' (choose from 'finetune')
```
→ Currently only `finetune` command is supported

### GPU Detection Issues

If automatic GPU detection fails, explicitly specify the number of processes:

```bash
automodel finetune llm -c config.yaml --nproc-per-node=1
```

## Advanced Usage

### Development and Debugging

For development workflows, you may want to use the Python scripts directly for additional debugging capabilities:

```bash
# Direct Python execution
uv run examples/llm/finetune.py --config examples/llm/llama_3_2_1b_squad.yaml

# With torchrun for multi-GPU
uv run torchrun --nproc-per-node=2 examples/llm/finetune.py \
    --config examples/llm/llama_3_2_1b_squad.yaml
```

### Container Integration

The CLI integrates with containerized environments and Slurm's container support:

```yaml
slurm:
  container_image: "nvcr.io/nvidia/nemo:dev"
  extra_mounts: "/data:/workspace/data"
```

## Getting Help

```bash
# Display CLI help
automodel --help

# Command-specific help
automodel finetune --help
```

## See Also

- {doc}`yaml-configuration-reference` - Complete YAML configuration options
- {doc}`../guides/launcher/slurm` - Slurm cluster setup and usage
- {doc}`../get-started/local-workstation` - Local development setup
