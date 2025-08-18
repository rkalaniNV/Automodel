# Build Multi-modal Models

Extend your skills to vision-language models by fine-tuning smaller VLMs on image-text datasets.

:::{note}
**Difficulty Level**: Intermediate  
**Estimated Time**: 45-60 minutes  
**Persona**: ML Engineers ready to explore multi-modal AI
:::

## Prerequisites

- Completed {doc}`first-fine-tuning`
- At least 12GB GPU memory (for smaller VLM)
- Understanding of multi-modal data concepts

## What You'll Learn

Building on your LLM fine-tuning knowledge, you'll master:

- **Multi-modal Concepts**: How vision and language models work together
- **Data Preparation**: Handling image-text paired datasets
- **VLM Architecture**: Understanding vision encoders and text decoders
- **Configuration Differences**: VLM-specific parameters and settings
- **Evaluation Techniques**: Measuring multi-modal model performance

## Understanding Vision-Language Models

**What are VLMs?** Vision-Language Models combine:
- **Vision Encoder**: Processes images into embeddings
- **Language Decoder**: Generates text based on visual and textual input
- **Cross-modal Fusion**: Connects visual and textual representations

**Use Cases:**
- Image captioning and description
- Visual question answering
- Document understanding
- Medical image analysis

## Step 1: Examine VLM Configuration

Look at the VLM training example:

```bash
cd examples/vlm
cat gemma_3_vl_2b_cord_v2.yaml
```

**Key VLM Configuration:**

```yaml
# Vision-Language Model (smaller for learning)
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: google/paligemma-3b-pt-448  # 3B parameter model
  torch_dtype: bfloat16  # Memory optimization

# Multi-modal dataset
data:
  _target_: nemo_automodel.datasets.vlm.CORD_v2  # Document understanding
  batch_size: 2
  image_size: 448
  max_seq_length: 512

# VLM-specific training
optimizer:
  _target_: torch.optim.AdamW
  lr: 1e-5  # Lower LR for VLM fine-tuning
  weight_decay: 0.01
```

## Step 2: Understanding Multi-modal Data

VLM training works with image-text pairs:

```python
# Example data format
{
  "image": "/path/to/medical_scan.jpg",
  "text": "This chest X-ray shows normal lung fields with no abnormalities."
}
```

## Step 3: Run VLM Training

```bash
# Launch VLM fine-tuning
automodel finetune vlm -c gemma_3_vl_2b_cord_v2.yaml
```

**VLM Training Process:**

1. **Image Processing**: Resizes and normalizes images
2. **Text Tokenization**: Processes captions/descriptions
3. **Multi-modal Fusion**: Combines vision and language features
4. **Supervised Learning**: Trains on image-text understanding tasks

## Step 4: Monitor VLM Metrics

VLM training shows additional metrics:

```text
[Step 50] Loss: 2.1 | Vision Loss: 0.8 | Language Loss: 1.3
[Step 100] Loss: 1.9 | Vision Loss: 0.7 | Language Loss: 1.2
```

## Step 5: Test Your VLM

```python
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

# Load fine-tuned VLM
model = AutoModelForCausalLM.from_pretrained("./checkpoints/final_model")
processor = AutoProcessor.from_pretrained("./checkpoints/final_model")

# Test with image
image = Image.open("test_medical_image.jpg")
prompt = "Describe what you see in this medical image:"

inputs = processor(text=prompt, images=image, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
description = processor.decode(outputs[0], skip_special_tokens=True)
print(description)
```

## Next Steps

- Try {doc}`parameter-efficient-fine-tuning` for memory-efficient training
- Explore custom datasets in {doc}`../../guides/vlm/dataset`

---

**Navigation:**
- ← {doc}`first-fine-tuning` Previous: Your First Fine-tuning Job
- ↑ {doc}`index` Back to Tutorials Overview
- → {doc}`parameter-efficient-fine-tuning` Next: Parameter-Efficient Fine-tuning
