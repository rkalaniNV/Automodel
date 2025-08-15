---
description: "Python API reference for NeMo Automodel core interfaces, model classes, and component APIs for programmatic usage"
tags: ["python-api", "interfaces", "classes", "methods", "programming"]
categories: ["reference"]
personas: ["mle-focused", "data-scientist-focused", "researcher-focused"]
difficulty: "reference"
content_type: "reference"
modality: "universal"
---

(api-interfaces-reference)=
# API Interfaces Reference

Python API reference for NeMo Automodel's core interfaces, model classes, and component APIs for programmatic usage.

## Overview

NeMo Automodel provides a rich Python API that enables programmatic access to all functionality available through YAML configurations. This reference covers the essential interfaces for AI developers building custom training workflows.

### Import Patterns

```python
# Core model classes
from nemo_automodel import NeMoAutoModelForCausalLM, NeMoAutoModelForImageTextToText

# Component imports
from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager
from nemo_automodel.components._peft.lora import PeftConfig
from nemo_automodel.components.datasets.llm.squad import make_squad_dataset
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
```

## Core Model Classes

### NeMoAutoModelForCausalLM

Drop-in replacement for Hugging Face's `AutoModelForCausalLM` with NeMo optimizations.

```python
class NeMoAutoModelForCausalLM:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        use_liger_kernel: bool = True,
        use_sdpa_patching: bool = True,
        sdpa_method: Optional[List[SDPBackend]] = None,
        torch_dtype: Union[str, torch.dtype] = "auto",
        attn_implementation: str = "flash_attention_2",
        fp8_config: Optional[object] = None,
        **kwargs,
    ) -> PreTrainedModel:
        """Load and optimize a language model for training."""
```

#### Usage Examples

**Basic Model Loading:**
```python
from nemo_automodel import NeMoAutoModelForCausalLM

# Load with default optimizations
model = NeMoAutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    torch_dtype="auto"
)

# Load with specific optimizations
model = NeMoAutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    torch_dtype=torch.bfloat16,
    use_liger_kernel=True,
    attn_implementation="flash_attention_2"
)
```

**Disable Optimizations:**
```python
# For debugging or compatibility
model = NeMoAutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    use_liger_kernel=False,
    use_sdpa_patching=False,
    attn_implementation="eager"
)
```

#### Key Parameters

```{list-table}
:header-rows: 1
:widths: 30 20 50

* - Parameter
  - Type
  - Description
* - `pretrained_model_name_or_path`
  - str
  - HuggingFace model ID or local path
* - `use_liger_kernel`
  - bool
  - Enable Liger optimized kernels (default: True)
* - `use_sdpa_patching`
  - bool
  - Apply SDPA attention patches (default: True)
* - `torch_dtype`
  - str/dtype
  - Model precision (`auto`, `torch.float16`, etc.)
* - `attn_implementation`
  - str
  - Attention backend (`flash_attention_2`, `eager`)
```

### NeMoAutoModelForImageTextToText

Optimized vision language model interface for multimodal training.

```python
class NeMoAutoModelForImageTextToText:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        use_liger_kernel: bool = True,
        use_sdpa_patching: bool = True,
        torch_dtype: Union[str, torch.dtype] = "auto",
        attn_implementation: str = "flash_attention_2",
        **kwargs,
    ) -> PreTrainedModel:
        """Load and optimize a vision language model."""
```

#### Usage Examples

```python
from nemo_automodel import NeMoAutoModelForImageTextToText

# Load VLM model
model = NeMoAutoModelForImageTextToText.from_pretrained(
    "google/gemma-3-4b-it",
    torch_dtype=torch.bfloat16,
    use_liger_kernel=False,  # Often disabled for VLMs
    attn_implementation="eager"
)
```

## Distributed Training Components

### FSDP2Manager

Fully Sharded Data Parallel manager for efficient multi-GPU training.

```python
class FSDP2Manager:
    def __init__(
        self,
        dp_size: Optional[int] = None,
        dp_replicate_size: int = 1,
        tp_size: int = 1,
        cp_size: int = 1,
        sequence_parallel: bool = False,
        **kwargs
    ):
        """Configure FSDP2 distributed training strategy."""
```

#### Usage Examples

```python
from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager

# Auto-detect data parallel size
dist_manager = FSDP2Manager(
    dp_size=None,  # Auto-detect
    tp_size=1,     # No tensor parallelism
    cp_size=1      # No context parallelism
)

# Explicit configuration
dist_manager = FSDP2Manager(
    dp_size=4,               # 4-way data parallelism
    dp_replicate_size=2,     # 2 replicas per shard
    tp_size=2,               # 2-way tensor parallelism
    sequence_parallel=True   # Enable sequence parallelism
)
```

### DDPManager

Simple Data Parallel manager for basic multi-GPU setups.

```python
from nemo_automodel.components.distributed.ddp import DDPManager

# Simple DDP setup
dist_manager = DDPManager()
```

## PEFT Components

### PeftConfig

Parameter-Efficient Fine-Tuning configuration for LoRA adapters.

```python
class PeftConfig:
    def __init__(
        self,
        match_all_linear: bool = False,
        include_modules: Optional[List[str]] = None,
        exclude_modules: Optional[List[str]] = None,
        dim: int = 8,
        alpha: int = 32,
        dropout: float = 0.0,
        use_triton: bool = True,
        **kwargs
    ):
        """Configure LoRA parameter-efficient fine-tuning."""
```

#### Usage Examples

```python
from nemo_automodel.components._peft.lora import PeftConfig

# Basic LoRA configuration
peft_config = PeftConfig(
    dim=8,           # LoRA rank
    alpha=32,        # Scaling factor
    dropout=0.1      # Adapter dropout
)

# Target specific modules
peft_config = PeftConfig(
    include_modules=[
        "*.q_proj",
        "*.v_proj", 
        "*.k_proj",
        "*.o_proj"
    ],
    exclude_modules=["*lm_head*"],
    dim=16,
    alpha=64
)

# VLM-specific configuration
vlm_peft_config = PeftConfig(
    exclude_modules=[
        "*vision_tower*",
        "*vision*",
        "*visual*",
        "*image_encoder*",
        "*lm_head*"
    ],
    dim=8,
    alpha=32,
    use_triton=True
)
```

## Dataset Components

### LLM Datasets

#### SQuAD Dataset

```python
def make_squad_dataset(
    dataset_name: str = "rajpurkar/squad",
    split: str = "train",
    limit_dataset_samples: Optional[int] = None,
    **kwargs
) -> torch.utils.data.Dataset:
    """Create SQuAD dataset for question-answering tasks."""
```

**Usage:**
```python
from nemo_automodel.components.datasets.llm.squad import make_squad_dataset

# Full training dataset
train_dataset = make_squad_dataset(
    dataset_name="rajpurkar/squad",
    split="train"
)

# Limited validation dataset
val_dataset = make_squad_dataset(
    dataset_name="rajpurkar/squad", 
    split="validation",
    limit_dataset_samples=100
)
```

### VLM Datasets

#### MedPix Dataset

```python
def make_medpix_dataset(
    path_or_dataset: str,
    split: str = "train",
    **kwargs
) -> torch.utils.data.Dataset:
    """Create MedPix dataset for medical image Q&A."""
```

**Usage:**
```python
from nemo_automodel.components.datasets.vlm.datasets import make_medpix_dataset

# Medical image Q&A dataset
dataset = make_medpix_dataset(
    path_or_dataset="mmoukouba/MedPix-VQA",
    split="train[:1000]"  # First 1000 samples
)
```

### Collate Functions

#### Default Collater

```python
def default_collater(batch):
    """Standard collate function for LLM datasets."""
```

#### VLM Collate Function

```python
def default_collate_fn(
    batch,
    start_of_response_token: str = "<start_of_turn>model\n",
    **kwargs
):
    """Collate function for vision language datasets."""
```

**Usage:**
```python
from torchdata.stateful_dataloader import StatefulDataLoader
from nemo_automodel.components.datasets.vlm.collate_fns import default_collate_fn

# VLM dataloader with custom tokens
dataloader = StatefulDataLoader(
    dataset,
    batch_size=1,
    collate_fn=lambda batch: default_collate_fn(
        batch, 
        start_of_response_token="<start_of_turn>model\n"
    )
)
```

## Loss Functions

### MaskedCrossEntropy

Memory-efficient cross-entropy loss with masking support.

```python
class MaskedCrossEntropy:
    def __init__(self, **kwargs):
        """Masked cross-entropy loss for language modeling."""
        
    def __call__(self, logits, targets, **kwargs):
        """Compute masked cross-entropy loss."""
```

**Usage:**
```python
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

loss_fn = MaskedCrossEntropy()
loss = loss_fn(model_logits, target_tokens)
```

### ChunkedCrossEntropy

Memory-optimized chunked cross-entropy for large vocabularies.

```python
from nemo_automodel.components.loss.chunked_ce import ChunkedCrossEntropy

loss_fn = ChunkedCrossEntropy(chunk_size=1024)
```

## Configuration Components

### ConfigNode

Dynamic configuration resolution system.

```python
from nemo_automodel.shared.config import ConfigNode

# Create configuration node
config = ConfigNode({
    "_target_": "torch.optim.Adam",
    "lr": 1e-5,
    "weight_decay": 0.01
})

# Instantiate object
optimizer = config.build(model.parameters())
```

### StatefulRNG

Reproducible random number generation for distributed training.

```python
from nemo_automodel.components.training.rng import StatefulRNG

# Ranked RNG (different seed per rank)
rng = StatefulRNG(seed=42, ranked=True)

# Global RNG (same seed all ranks)  
rng = StatefulRNG(seed=42, ranked=False)
```

## Training Utilities

### Checkpoint Management

```python
from nemo_automodel.components.checkpoint import CheckpointManager

checkpoint_manager = CheckpointManager(
    checkpoint_dir="./checkpoints",
    model_save_format="safetensors"
)

# Save checkpoint
checkpoint_manager.save(model, optimizer, step=100)

# Load checkpoint
checkpoint_manager.load(model, optimizer)
```

### Model Freezing

```python
def apply_freeze_config(model, freeze_config):
    """Apply parameter freezing configuration to model."""
    
# Freeze configuration
freeze_config = {
    "freeze_embeddings": True,
    "freeze_vision_tower": True,
    "freeze_language_model": False
}

apply_freeze_config(model, freeze_config)
```

## Error Handling

### UnavailableError

```python
from nemo_automodel.shared.import_utils import UnavailableError

try:
    from optional_dependency import OptionalClass
except ImportError:
    raise UnavailableError("optional_dependency not installed")
```

### Common Exception Patterns

```python
# Model loading with fallbacks
try:
    model = NeMoAutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2"
    )
except ValueError as e:
    if "does not support" in str(e):
        # Fallback to eager attention
        model = NeMoAutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="eager"
        )
```

## Integration Examples

### Complete Training Loop

```python
import torch
from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager
from nemo_automodel.components.datasets.llm.squad import make_squad_dataset
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

# Initialize components
model = NeMoAutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
dist_manager = FSDP2Manager(dp_size=None, tp_size=1)
dataset = make_squad_dataset("rajpurkar/squad", "train")
loss_fn = MaskedCrossEntropy()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Apply distributed strategy
model = dist_manager.wrap_model(model)

# Training loop
for batch in dataset:
    optimizer.zero_grad()
    logits = model(batch["input_ids"])
    loss = loss_fn(logits, batch["labels"])
    loss.backward()
    optimizer.step()
```

### PEFT Training

```python
from nemo_automodel.components._peft.lora import PeftConfig

# Setup PEFT
peft_config = PeftConfig(dim=8, alpha=32)
model = peft_config.apply(model)

# Only adapter parameters are trainable
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
```

## Advanced Usage

### Custom Components

Extend NeMo Automodel with custom components:

```python
class CustomDataset:
    def __init__(self, data_path: str, **kwargs):
        self.data = self.load_data(data_path)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)

# Use in YAML configuration
# dataset:
#   _target_: my_module.CustomDataset
#   data_path: /path/to/data
```

### Dynamic Configuration

```python
from nemo_automodel.shared.config import ConfigNode

# Runtime configuration modification
config = ConfigNode.from_yaml("config.yaml")
config.optimizer.lr = 2e-5  # Override learning rate
config.model.torch_dtype = torch.float16  # Change precision

# Build components
model = config.model.build()
optimizer = config.optimizer.build(model.parameters())
```

## See Also

- {doc}`yaml-configuration-reference` - YAML configuration schemas
- {doc}`cli-command-reference` - Command-line interface
- {doc}`../api-docs/index` - Complete API documentation
- {doc}`troubleshooting-reference` - Error handling and debugging
