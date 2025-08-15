# {py:mod}`datasets.vlm.collate_fns`

```{py:module} datasets.vlm.collate_fns
```

```{autodoc2-docstring} datasets.vlm.collate_fns
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_loss_mask_with_start_of_response_token <datasets.vlm.collate_fns.create_loss_mask_with_start_of_response_token>`
  - ```{autodoc2-docstring} datasets.vlm.collate_fns.create_loss_mask_with_start_of_response_token
    :summary:
    ```
* - {py:obj}`phi4_mm_collate_fn <datasets.vlm.collate_fns.phi4_mm_collate_fn>`
  - ```{autodoc2-docstring} datasets.vlm.collate_fns.phi4_mm_collate_fn
    :summary:
    ```
* - {py:obj}`qwen2_5_collate_fn <datasets.vlm.collate_fns.qwen2_5_collate_fn>`
  - ```{autodoc2-docstring} datasets.vlm.collate_fns.qwen2_5_collate_fn
    :summary:
    ```
* - {py:obj}`default_collate_fn <datasets.vlm.collate_fns.default_collate_fn>`
  - ```{autodoc2-docstring} datasets.vlm.collate_fns.default_collate_fn
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`COLLATE_FNS <datasets.vlm.collate_fns.COLLATE_FNS>`
  - ```{autodoc2-docstring} datasets.vlm.collate_fns.COLLATE_FNS
    :summary:
    ```
````

### API

````{py:function} create_loss_mask_with_start_of_response_token(input_ids, processor, start_of_response_token=None)
:canonical: datasets.vlm.collate_fns.create_loss_mask_with_start_of_response_token

```{autodoc2-docstring} datasets.vlm.collate_fns.create_loss_mask_with_start_of_response_token
```
````

````{py:function} phi4_mm_collate_fn(examples, processor)
:canonical: datasets.vlm.collate_fns.phi4_mm_collate_fn

```{autodoc2-docstring} datasets.vlm.collate_fns.phi4_mm_collate_fn
```
````

````{py:function} qwen2_5_collate_fn(examples: list, processor, start_of_response_token='<|im_start|>assistant\n') -> dict[str, torch.Tensor]
:canonical: datasets.vlm.collate_fns.qwen2_5_collate_fn

```{autodoc2-docstring} datasets.vlm.collate_fns.qwen2_5_collate_fn
```
````

````{py:function} default_collate_fn(examples: list, processor, start_of_response_token=None) -> dict[str, torch.Tensor]
:canonical: datasets.vlm.collate_fns.default_collate_fn

```{autodoc2-docstring} datasets.vlm.collate_fns.default_collate_fn
```
````

````{py:data} COLLATE_FNS
:canonical: datasets.vlm.collate_fns.COLLATE_FNS
:value: >
   None

```{autodoc2-docstring} datasets.vlm.collate_fns.COLLATE_FNS
```

````
