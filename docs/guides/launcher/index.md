---
description: "Learn how to launch NeMo Automodel training jobs across different environments including local workstations and SLURM clusters."
tags: ["launcher", "slurm", "distributed", "cluster"]
categories: ["deployment"]
---

(launcher-guide)=
# Launch Training Jobs

Learn how to launch NeMo Automodel training jobs across different computing environments, from single GPUs to multi-node clusters.

(launcher-overview)=
## Overview

The NeMo Automodel launcher provides a unified interface for running training jobs across various environments. Whether you're working on a local workstation, a SLURM cluster, or cloud infrastructure, the launcher handles the complexity of distributed training setup.

(launcher-supported-environments)=
## Supported Environments

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` SLURM Clusters
:link: slurm
:link-type: doc
:link-alt: SLURM launcher guide

Launch distributed training jobs on SLURM-managed clusters with automatic resource allocation and job scheduling.
+++
{bdg-primary}`SLURM`
{bdg-secondary}`HPC`
:::

:::{grid-item-card} {octicon}`desktop-download;1.5em;sd-mr-1` Local Workstations
:link: ../../get-started/local-workstation
:link-type: doc
:link-alt: Local workstation setup

Run training jobs on local workstations with single or multiple GPUs using simple CLI commands.
+++
{bdg-info}`Local`
{bdg-secondary}`Single Node`
:::

::::

(launcher-key-features)=
## Key Features

- **Unified CLI**: Single command interface across all environments
- **Automatic Scaling**: Seamless scaling from single GPU to multi-node clusters
- **Resource Management**: Intelligent resource allocation and job scheduling
- **Container Support**: Integration with NVIDIA containers and custom images
- **Flexible Configuration**: YAML-based configuration for all deployment scenarios

(launcher-getting-started)=
## Get Started

1. **Configure your environment** with the appropriate settings for your cluster or workstation
2. **Prepare your YAML configuration** with model, data, and training parameters
3. **Launch your job** using the `automodel` CLI with your target environment
4. **Monitor progress** through built-in logging and checkpoint management

(launcher-environment-setup)=
## Environment-Specific Setup

Each environment requires specific configuration:

- **SLURM**: Configure job parameters, partition settings, and container images
- **Local**: Set up GPU devices and ensure proper driver installation
- **Cloud**: Configure cloud provider credentials and instance types

```{toctree}
:maxdepth: 2
:titlesonly:
:hidden:

slurm
```