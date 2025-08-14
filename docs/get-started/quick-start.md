(get-started-quick-start)=
# Quick Start

Follow these steps to fine-tune a Hugging Face model with NeMo Automodel.

## Prerequisites

- Python 3.10+
- NVIDIA GPU with a recent CUDA toolkit
- PyTorch with CUDA installed

## Steps

1. Instantiate a Hugging Face model with Automodel
2. Prepare your dataset (HF `datasets` works well)
3. Choose PEFT (e.g., LoRA) or full-parameter SFT
4. Configure parallelism (DDP or FSDP2) and training hyperparameters
5. Launch training and export for inference when done

## Run a Recipe with torchrun

```bash
torchrun --nproc-per-node=2 nemo_automodel/recipes/llm/finetune.py -c examples/llm/llama_3_2_1b_squad.yaml
```

## Or use the automodel CLI

```bash
automodel llm finetune -c examples/llm/llama_3_2_1b_squad.yaml --nproc-per-node=2
```

## Example: LoRA or full-parameter SFT

::: {dropdown} Python example
:icon: code-square

```python
from datasets import load_dataset

# Prepare data
dataset = load_dataset("rajpurkar/squad", split="train")
dataset = dataset.map(formatting_prompts_func)

# Launch a fine-tuning run (illustrative API)
llm.api.finetune(
    # Model & PEFT scheme
    model=llm.HFAutoModelForCausalLM(model_id),

    # Setting peft=None will run full-parameter SFT
    peft=llm.peft.LoRA(
        target_modules=["*_proj", "linear_qkv"],  # regex-based selector
        dim=32,
    ),

    # Data
    data=llm.HFDatasetDataModule(dataset),

    # Optimizer
    optim=fdl.build(llm.adam.pytorch_adam_with_flat_lr(lr=1e-5)),

    # Trainer
    trainer=nl.Trainer(
        devices=args.devices,
        max_steps=args.max_steps,
        strategy=args.strategy,  # choices: [None, "ddp", FSDP2Strategy]
    ),
)
```
:::

## Switch to Megatron-Core for peak throughput

Minimal changes enable the high-throughput Megatron-Core backend when available:

::: {dropdown} Code diff (illustrative)
:icon: code-square

```python
# Model class
# Automodel
model = llm.HFAutoModelForCausalLM(model_id)
# Megatron-Core
model = llm.LlamaModel(Llama32Config1B())

# Optimizer module
# Automodel
optim = fdl.build(llm.adam.pytorch_adam_with_flat_lr(lr=1e-5))
# Megatron-Core
optim = MegatronOptimizerModule(config=opt_config)

# Trainer strategy
# Automodel
strategy = args.strategy  # [None, "ddp", "fsdp2"]
# Megatron-Core
strategy = nl.MegatronStrategy(ddp="pytorch")
```
:::

## Next steps

- Read the {ref}`about-overview` for architecture and context
- Explore {ref}`about-key-features` for capabilities and performance notes

## Extending Automodel

Today, Automodel supports `AutoModelForCausalLM` for text generation. To add support for another task (for example, Seq2Seq LM), create a subclass similar to `HFAutoModelForCausalLM` and adapt initialization, configuration, training/validation steps, save/load routines, checkpoint handling, and provide a compatible data module for your dataset and batching needs.
