# {py:mod}`nemo_automodel.components.datasets.vlm.utils`

```{py:module} nemo_automodel.components.datasets.vlm.utils
```

```{autodoc2-docstring} nemo_automodel.components.datasets.vlm.utils
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`extract_skipped_token_ids <nemo_automodel.components.datasets.vlm.utils.extract_skipped_token_ids>`
  - ```{autodoc2-docstring} nemo_automodel.components.datasets.vlm.utils.extract_skipped_token_ids
    :summary:
    ```
* - {py:obj}`json2token <nemo_automodel.components.datasets.vlm.utils.json2token>`
  - ```{autodoc2-docstring} nemo_automodel.components.datasets.vlm.utils.json2token
    :summary:
    ```
* - {py:obj}`process_text_batch <nemo_automodel.components.datasets.vlm.utils.process_text_batch>`
  - ```{autodoc2-docstring} nemo_automodel.components.datasets.vlm.utils.process_text_batch
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`QWEN_TOKENS <nemo_automodel.components.datasets.vlm.utils.QWEN_TOKENS>`
  - ```{autodoc2-docstring} nemo_automodel.components.datasets.vlm.utils.QWEN_TOKENS
    :summary:
    ```
* - {py:obj}`LLAVA_TOKENS <nemo_automodel.components.datasets.vlm.utils.LLAVA_TOKENS>`
  - ```{autodoc2-docstring} nemo_automodel.components.datasets.vlm.utils.LLAVA_TOKENS
    :summary:
    ```
* - {py:obj}`LLAMA_TOKENS <nemo_automodel.components.datasets.vlm.utils.LLAMA_TOKENS>`
  - ```{autodoc2-docstring} nemo_automodel.components.datasets.vlm.utils.LLAMA_TOKENS
    :summary:
    ```
* - {py:obj}`GEMMA_TOKENS <nemo_automodel.components.datasets.vlm.utils.GEMMA_TOKENS>`
  - ```{autodoc2-docstring} nemo_automodel.components.datasets.vlm.utils.GEMMA_TOKENS
    :summary:
    ```
* - {py:obj}`GEMMA_3N_TOKENS <nemo_automodel.components.datasets.vlm.utils.GEMMA_3N_TOKENS>`
  - ```{autodoc2-docstring} nemo_automodel.components.datasets.vlm.utils.GEMMA_3N_TOKENS
    :summary:
    ```
* - {py:obj}`PAD_TOKENS <nemo_automodel.components.datasets.vlm.utils.PAD_TOKENS>`
  - ```{autodoc2-docstring} nemo_automodel.components.datasets.vlm.utils.PAD_TOKENS
    :summary:
    ```
````

### API

````{py:data} QWEN_TOKENS
:canonical: nemo_automodel.components.datasets.vlm.utils.QWEN_TOKENS
:value: >
   ['<|im_start|>', '<|im_end|>', '<|vision_start|>', '<|vision_end|>', '<|vision_pad|>', '<|image_pad|...

```{autodoc2-docstring} nemo_automodel.components.datasets.vlm.utils.QWEN_TOKENS
```

````

````{py:data} LLAVA_TOKENS
:canonical: nemo_automodel.components.datasets.vlm.utils.LLAVA_TOKENS
:value: >
   ['<image>', '<pad>']

```{autodoc2-docstring} nemo_automodel.components.datasets.vlm.utils.LLAVA_TOKENS
```

````

````{py:data} LLAMA_TOKENS
:canonical: nemo_automodel.components.datasets.vlm.utils.LLAMA_TOKENS
:value: >
   ['<|begin_of_text|>', '<|end_of_text|>', '<|finetune_right_pad_id|>', '<|step_id|>', '<|start_header...

```{autodoc2-docstring} nemo_automodel.components.datasets.vlm.utils.LLAMA_TOKENS
```

````

````{py:data} GEMMA_TOKENS
:canonical: nemo_automodel.components.datasets.vlm.utils.GEMMA_TOKENS
:value: >
   ['<image_soft_token>']

```{autodoc2-docstring} nemo_automodel.components.datasets.vlm.utils.GEMMA_TOKENS
```

````

````{py:data} GEMMA_3N_TOKENS
:canonical: nemo_automodel.components.datasets.vlm.utils.GEMMA_3N_TOKENS
:value: >
   ['<image_soft_token>', '<audio_soft_token>', '<start_of_audio>', '<start_of_image>', '<end_of_audio>...

```{autodoc2-docstring} nemo_automodel.components.datasets.vlm.utils.GEMMA_3N_TOKENS
```

````

````{py:data} PAD_TOKENS
:canonical: nemo_automodel.components.datasets.vlm.utils.PAD_TOKENS
:value: >
   'set(...)'

```{autodoc2-docstring} nemo_automodel.components.datasets.vlm.utils.PAD_TOKENS
```

````

````{py:function} extract_skipped_token_ids(processor)
:canonical: nemo_automodel.components.datasets.vlm.utils.extract_skipped_token_ids

```{autodoc2-docstring} nemo_automodel.components.datasets.vlm.utils.extract_skipped_token_ids
```
````

````{py:function} json2token(obj, sort_json_key: bool = True)
:canonical: nemo_automodel.components.datasets.vlm.utils.json2token

```{autodoc2-docstring} nemo_automodel.components.datasets.vlm.utils.json2token
```
````

````{py:function} process_text_batch(processor, texts: list[str], images: list | None = None) -> dict[str, torch.Tensor]
:canonical: nemo_automodel.components.datasets.vlm.utils.process_text_batch

```{autodoc2-docstring} nemo_automodel.components.datasets.vlm.utils.process_text_batch
```
````
