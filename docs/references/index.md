---
author: lawrence lane
description: "Access comprehensive reference documentation including API specifications, configuration options, and technical details."
tags: ["reference", "api", "configuration", "specifications"]
categories: ["reference", "onboarding"]
---

(ref-overview)=
# About NeMo Automodel References

Comprehensive reference documentation for NeMo Automodel including API specifications, configuration options, command-line interfaces, and technical details.

## Overview

Our reference documentation provides detailed technical information for developers, researchers, and engineers working with NeMo Automodel. Whether you're implementing custom training workflows, debugging issues, or exploring advanced configuration options, these references provide the authoritative information you need.

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` Command-Line Interface
:link: cli-command-reference
:link-type: doc
:link-alt: Complete CLI reference

Complete reference for the `automodel` CLI with all commands, options, and usage patterns.
+++
{bdg-primary}`CLI` {bdg-secondary}`Commands`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` YAML Configuration
:link: yaml-configuration-reference  
:link-type: doc
:link-alt: Configuration schema reference

Comprehensive YAML configuration schema with all parameters, sections, and examples.
+++
{bdg-primary}`YAML` {bdg-secondary}`Configuration`
:::

::::

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Python API
:link: api-interfaces-reference
:link-type: doc  
:link-alt: Python API interfaces

Python API reference covering core classes, methods, and programmatic interfaces.
+++
{bdg-primary}`Python` {bdg-secondary}`API`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Troubleshooting
:link: troubleshooting-reference
:link-type: doc
:link-alt: Error handling and debugging

Comprehensive troubleshooting guide with common errors, solutions, and debugging strategies.
+++  
{bdg-primary}`Debug` {bdg-secondary}`Support`
:::

::::

## Reference Categories

### For AI Developers

**Essential references for machine learning engineers and researchers:**

- **{doc}`cli-command-reference`** - Master the `automodel` CLI for efficient workflow management
- **{doc}`yaml-configuration-reference`** - Understand all configuration options for training customization
- **{doc}`api-interfaces-reference`** - Access programmatic interfaces for custom implementations

### For DevOps and Infrastructure

**Technical references for deployment and operations:**

- **{doc}`troubleshooting-reference`** - Diagnose and resolve common deployment issues
- **Environment Variables** - Configuration through environment settings
- **Container Integration** - Docker and Slurm container setup

### For Enterprise Users

**Advanced configuration and scaling references:**

- **Distributed Training** - Multi-node and cluster configuration
- **Security Configuration** - Authentication and access control
- **Performance Optimization** - Advanced tuning and monitoring

## Quick Reference Links

### Most Used References

```{list-table}
:header-rows: 1
:widths: 40 60

* - Reference
  - Use Case
* - {doc}`cli-command-reference`
  - Daily CLI usage and automation
* - {doc}`yaml-configuration-reference`  
  - Training configuration and experimentation
* - {doc}`troubleshooting-reference`
  - Issue resolution and debugging
* - {doc}`api-interfaces-reference`
  - Custom development and integration
```

### Configuration Quick Start

**Common configuration patterns:**

1. **Single GPU training:** Basic YAML setup for local development
2. **Multi-GPU training:** FSDP2 configuration for distributed training  
3. **PEFT fine-tuning:** LoRA configuration for efficient adaptation
4. **Slurm clusters:** Batch job configuration for HPC environments

## Additional Resources

### Related Documentation

- **{doc}`../guides/index`** - Step-by-step training guides
- **{doc}`../api-docs/index`** - Complete API documentation
- **{doc}`../about/architecture-overview`** - System architecture and design

### External Resources

- **[Hugging Face Documentation](https://huggingface.co/docs)** - Model and dataset references
- **[PyTorch Documentation](https://pytorch.org/docs)** - Framework-specific details
- **[NVIDIA Developer Documentation](https://developer.nvidia.com/)** - GPU optimization guides

## Contributing to References

Help improve our reference documentation:

1. **Report Issues** - Submit feedback for unclear or missing information
2. **Submit Examples** - Contribute working configuration examples
3. **Technical Reviews** - Validate accuracy of technical details

For contribution guidelines, see our [Documentation Standards](../about/repository-and-package-guide.md#documentation-standards).
