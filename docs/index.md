---
description: "Explore comprehensive documentation for our software platform, including tutorials, feature guides, and deployment instructions."
tags: ["overview", "quickstart", "getting-started"]
categories: ["getting-started"]
---

(template-home)=

# NeMo Automodel Documentation

Welcome. This site helps AI developers fine-tune and scale Hugging Face models using NVIDIA NeMo Automodel.

## What is NeMo Automodel?

High-level overview and links:

- About: introduction, architecture, and why Day‑0 matters
- Key Features: capabilities, backends, and performance notes
- Quick Start: install and run your first fine-tune

## Quick Start

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Feature Set A
:link: feature-set-a
:link-type: ref
:link-alt: Feature Set A documentation home

Comprehensive tools and workflows for data processing and analysis.
Get started with our core feature set.
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Feature Set B  
:link: feature-set-b
:only: not ga
:link-type: ref
:link-alt: Feature Set B documentation home

Advanced integration capabilities and specialized processing tools.
Available in Early Access.
:::

::::

## Core Learning Path

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Feature Set A Tutorials
:link: feature-set-a-tutorials
:link-type: ref
:link-alt: Feature Set A tutorial collection

Step-by-step guides for getting the most out of Feature Set A
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Feature Set B Tutorials
:link: feature-set-b-tutorials
:only: not ga
:link-type: ref
:link-alt: Feature Set B tutorial collection

Hands-on tutorials for Feature Set B workflows
:::

::::

## Key Guides

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Deployment Patterns
:link: admin-deployment
:link-type: ref
:link-alt: Deployment and configuration guides

Learn how to deploy and configure your environment
:::

:::{grid-item-card} {octicon}`plug;1.5em;sd-mr-1` Integration Patterns
:link: admin-integrations
:link-type: ref
:link-alt: Integration and connection guides

Connect with external systems and services
:::

::::

---

::::{toctree}
:hidden:
Home <self>
::::

::::{toctree}
:hidden:
:caption: About 
:maxdepth: 1
about/index.md
about/key-features.md
about/concepts/index.md
about/release-notes/index.md
::::

::::{toctree}
:hidden:
:caption: Get Started
:maxdepth: 2

get-started/index.md
Feature Set A Quickstart <get-started/feature-set-a.md>
Feature Set B Quickstart <get-started/feature-set-b.md> :only: not ga
::::

::::{toctree}
:hidden:
:caption: (GA) Feature Set A
:maxdepth: 2
feature-set-a/index.md
Tutorials <feature-set-a/tutorials/index.md>
feature-set-a/category-a/index.md
::::

::::{toctree}
:hidden:
:caption: (EA) Feature Set B
:maxdepth: 2
:only: not ga 

feature-set-b/index.md
Tutorials <feature-set-b/tutorials/index.md>
feature-set-b/category-a/index.md
::::

::::{toctree}
:hidden:
:caption: Admin
:maxdepth: 2
admin/index.md
Deployment <admin/deployment/index.md>
Integrations <admin/integrations/index.md>
CI/CD <admin/cicd/index.md>
::::

::::{toctree}
:hidden:
:caption: API Reference
:maxdepth: 2

api-docs/index.md
api-docs/components/index.md
api-docs/recipes/index.md
api-docs/cli/index.md
api-docs/shared/index.md
::::

::::{toctree}
:hidden:
:caption: Reference
:maxdepth: 2
references/index.md
::::

## Reference Documentation

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration Reference
:link: references/configuration-reference
:link-type: doc

Complete reference for all configuration options and parameters.

+++
{bdg-primary}`Reference`
:::

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` CLI Reference
:link: references/cli-reference
:link-type: doc

Command-line interface commands and usage patterns.

+++
{bdg-info}`CLI`
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` API Documentation
   :link: api-docs/index
:link-type: doc

Comprehensive API documentation for all NeMo RL components.

+++
{bdg-warning}`Development`
:::

:::{grid-item-card} {octicon}`light-bulb;1.5em;sd-mr-1` Core Design and Architecture
:link: core-design/index
:link-type: doc

Architectural decisions and technical specifications for framework internals.

+++
{bdg-warning}`Advanced`
:::

::::

## Research Methodologies

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Experimental Design
:link: advanced/research/experimental-design-validation
:link-type: doc

Design controlled experiments and research studies with proper experimental methodology for NeMo RL research.

+++
{bdg-info}`Research Methodology`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Model Evaluation
:link: advanced/research/model-evaluation-validation
:link-type: doc

Build comprehensive evaluation frameworks and implement robust model assessment and comparison strategies.

+++
{bdg-info}`Evaluation Framework`
:::

:::{grid-item-card} {octicon}`chart;1.5em;sd-mr-1` Performance Analysis
:link: advanced/research/performance-analysis
:link-type: doc

Analyze model performance and interpret results with statistical rigor and comprehensive metrics.

+++
{bdg-info}`Performance Analysis`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Custom Algorithms
:link: advanced/research/custom-algorithms
:link-type: doc

Develop custom algorithms and extend NeMo RL with new training approaches and methodologies.

+++
{bdg-info}`Algorithm Development`
:::

:::{grid-item-card} {octicon}`search;1.5em;sd-mr-1` Ablation Studies
:link: advanced/research/ablation-studies
:link-type: doc

Conduct systematic ablation studies to understand model components and their contributions.

+++
{bdg-info}`Component Analysis`
:::

:::{grid-item-card} {octicon}`check-circle;1.5em;sd-mr-1` Reproducible Research
:link: advanced/research/reproducible-research-validation
:link-type: doc

Implement deterministic training and environment management for reproducible experiments.

+++
{bdg-info}`Reproducibility`
:::

::::

## Getting Help

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`bug;1.5em;sd-mr-1` Troubleshooting
:link: guides/troubleshooting
:link-type: doc

Common issues, error messages, and solutions.

+++
{bdg-warning}`Support`
:::

:::{grid-item-card} {octicon}`question;1.5em;sd-mr-1` Production Support
:link: guides/troubleshooting
:link-type: doc

Deployment guides, monitoring, and production best practices.

+++
{bdg-info}`Production`
:::

::::

---