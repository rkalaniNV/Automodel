# NeMo Framework Now Supports Google Gemma 3n: Efficient Multimodal Fine-Tuning Made Simple

## Introduction

[Gemma 3n](https://ai.google.dev/gemma/docs/gemma-3n) is a generative AI model that takes inputs from a variety of modalities, including images and audio, and is optimized for efficient resource usage and fast inference on everyday devices. It introduces innovations such as Per-Layer Embedding parameter caching and the [MatFormer](https://arxiv.org/pdf/2310.07707) architecture, which help reduce compute and memory demands, making it ideal for lightweight deployments. Some key highlights:

- **Optimized architecture** featuring MatFormer's nested transformers, Per-Layer Embeddings (PLE), and KV cache sharing. These enable sub-model extraction, reduced GPU memory usage, and faster prefill speeds. For more details, check out [documentation](https://ai.google.dev/gemma/docs/gemma-3n).

- **Multimodal capabilities** with integrated image and audio encoders alongside the language model, enabling diverse tasks across modalities.
- Pretrained checkpoints are available under the [Gemma 3n releases on Hugging Face](https://huggingface.co/collections/google/gemma-3n-685065323f5984ef315c93f4).

Today, we are excited to announce that NeMo Automodel now supports Gemma 3n, making it easier than ever to load, train, and inference with Gemma 3n models.


---

## Fine-tuning Gemma 3n with NeMo Automodel

[NeMo Framework's Automodel path ("Nemo AutoModel")](https://github.com/NVIDIA-NeMo/Automodel) offers day-0 support for :hugs:Hugging Face models via a unified interface to load and finetune models across modalities, abstracting away backend complexity. With Gemma 3n support:

- Load models with a single `from_pretrained` call
- Fine-tune models using full parameter training or PEFT (LoRA) with predefined recipes
- Accelerate training with kernel optimizations
- Leverage FSDP2/nvFSDP for efficient distributed training

Check out our [tutorial](https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/guides/omni/gemma3-3n.md) on SFT and PEFT for both Gemma 3 and Gemma 3n models!

## Observations

### Training Dynamics
During the first hundred optimization steps we observed suspiciously large gradients.
However, after a few iterations it quickly stabilizes. While the run remains numerically stable after this "warm-up," overall convergence still lags behind Gemma 3. We continue to investigate the source of this discrepancy.

```{raw} html
<div id="blog-training-loss-modal-container" style="cursor: pointer; display: inline-block; border: 2px solid transparent; border-radius: 8px; transition: border-color 0.3s ease;" 
     onmouseover="this.style.borderColor='#007acc'" 
     onmouseout="this.style.borderColor='transparent'"
     onclick="openImageModal('blog-training-loss-modal', 'omni/medpix_peft.jpg', 'Training Loss Curve')">
```

```{image} omni/medpix_peft.jpg
:alt: Training Loss Curve.
:class: bg-primary
:width: 400px
:align: center
```

```{raw} html
</div>

<!-- Modal for Blog Training Loss -->
<div id="blog-training-loss-modal" style="display: none; position: fixed; z-index: 10000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.8); backdrop-filter: blur(3px);" onclick="closeImageModal('blog-training-loss-modal')">
    <div style="position: relative; margin: auto; padding: 20px; width: 90%; max-width: 1200px; top: 50%; transform: translateY(-50%);">
        <span onclick="closeImageModal('blog-training-loss-modal')" style="color: white; float: right; font-size: 28px; font-weight: bold; cursor: pointer; background: rgba(0,0,0,0.5); padding: 5px 10px; border-radius: 50%;">&times;</span>
        <div style="text-align: center; max-height: 80vh; overflow-y: auto;">
            <img id="blog-training-loss-modal-img" src="" alt="" style="max-width: 100%; height: auto; border-radius: 8px;">
            <p id="blog-training-loss-modal-caption" style="color: white; margin-top: 15px; font-size: 16px;"></p>
        </div>
    </div>
</div>

<script>
function openImageModal(modalId, imgSrc, caption) {
    const modal = document.getElementById(modalId);
    const modalImg = document.getElementById(modalId + '-img');
    const modalCaption = document.getElementById(modalId + '-caption');
    
    modal.style.display = 'block';
    modalImg.src = imgSrc;
    modalCaption.textContent = caption;
    
    // Prevent body scroll when modal is open
    document.body.style.overflow = 'hidden';
}

function closeImageModal(modalId) {
    const modal = document.getElementById(modalId);
    modal.style.display = 'none';
    document.body.style.overflow = 'auto';
}

// Close modal with Escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const modals = document.querySelectorAll('[id*="-modal"]');
        modals.forEach(modal => {
            if (modal.style.display === 'block') {
                modal.style.display = 'none';
                document.body.style.overflow = 'auto';
            }
        });
    }
});
</script>
```

Our preliminary benchmark on vision and audio capabilities shows some gaps between Gemma 3n and existing alternatives. We will follow up with more concrete results later.


## Conclusion
Gemma 3n brings impressive efficiency and opens up new possibilities for multimodal tasks on devices. With NeMo Automodel, getting started and fine-tuning these efficient models requires only a few commands!

We look forward to seeing what you build with Gemma 3n and NeMo Automodel. Check out the documentation guide for a full walkthrough, and reach out on GitHub Discussions if you have questions.

## References
[Gemma 3n on Hugging Face](https://huggingface.co/collections/google/gemma-3n-685065323f5984ef315c93f4)

[NeMo Automodel](https://github.com/NVIDIA-NeMo/Automodel)

[NeMo Automodel Gemma 3 Guide](https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/guides/omni/gemma3-3n.md)
