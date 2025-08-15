# Examples

Ready-to-run examples and configurations demonstrating NeMo Automodel capabilities across different model types and training scenarios.

## Overview

Our examples provide working YAML configurations and Python scripts that you can use immediately to start training with NeMo Automodel. Each example includes complete setup, training, and evaluation workflows.

## Example Categories

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`comment-discussion;1.5em;sd-mr-1` LLM Examples
:link: /examples/llm/
:link-type: url
:link-alt: LLM training examples

Complete examples for fine-tuning large language models including Llama, Mistral, Gemma, and more.
+++
{bdg-primary}`LLMs`
{bdg-secondary}`Text Generation`
:::

:::{grid-item-card} {octicon}`image;1.5em;sd-mr-1` VLM Examples
:link: /examples/vlm/
:link-type: url
:link-alt: VLM training examples

Examples for training vision language models with multimodal datasets and specialized preprocessing.
+++
{bdg-info}`VLMs`
{bdg-secondary}`Multimodal`
:::

::::

## Featured Examples

### Language Model Training
- **Llama 3.2 1B SFT**: Full supervised fine-tuning on instruction data
- **Llama 3.2 1B PEFT**: Parameter-efficient fine-tuning with LoRA
- **Qwen 0.6B HellaSwag**: Evaluation benchmark training
- **Squad QA Training**: Question-answering fine-tuning

### Vision Language Models
- **Gemma 3 VL CORD-V2**: Document understanding with PEFT
- **Gemma 3 VL MedPix**: Medical visual question answering
- **Gemma 3n VL Training**: Omni-modal model fine-tuning
- **Phi 4 MM Training**: Multi-modal conversation training

### Distributed Training
- **nvFSDP Scaling**: Multi-GPU training with NVIDIA's optimized FSDP
- **FSDP2 Examples**: Distributed training with PyTorch FSDP2
- **FP8 Quantization**: Memory-efficient training with FP8

## How to Use Examples

1. **Navigate to the examples directory** in your NeMo Automodel installation
2. **Choose an example** that matches your model type and use case
3. **Review the YAML configuration** to understand the settings
4. **Run the example** using the provided Python scripts
5. **Modify settings** to adapt the example to your specific needs

## Example Structure

Each example typically includes:
- **YAML Configuration**: Complete training configuration
- **Python Script**: Execution script with proper imports
- **Documentation**: Explanation of settings and expected outcomes
- **Requirements**: Specific dependencies and hardware requirements

## Getting Started

For first-time users, we recommend starting with:

1. **LLM SFT Example**: `examples/llm/llama_3_2_1b_squad.yaml`
2. **PEFT Example**: `examples/llm/llama_3_2_1b_hellaswag_peft.yaml`
3. **VLM Example**: `examples/vlm/gemma_3_vl_4b_medpix_peft.yaml`

These examples provide a solid foundation for understanding NeMo Automodel's capabilities and can be easily adapted for your specific requirements.

```{toctree}
:maxdepth: 2
:titlesonly:
:hidden:

beginner
```
