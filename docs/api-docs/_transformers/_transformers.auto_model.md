# {py:mod}`_transformers.auto_model`

```{py:module} _transformers.auto_model
```

```{autodoc2-docstring} _transformers.auto_model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_BaseNeMoAutoModelClass <_transformers.auto_model._BaseNeMoAutoModelClass>`
  - ```{autodoc2-docstring} _transformers.auto_model._BaseNeMoAutoModelClass
    :summary:
    ```
* - {py:obj}`NeMoAutoModelForCausalLM <_transformers.auto_model.NeMoAutoModelForCausalLM>`
  - ```{autodoc2-docstring} _transformers.auto_model.NeMoAutoModelForCausalLM
    :summary:
    ```
* - {py:obj}`NeMoAutoModelForImageTextToText <_transformers.auto_model.NeMoAutoModelForImageTextToText>`
  - ```{autodoc2-docstring} _transformers.auto_model.NeMoAutoModelForImageTextToText
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_assert_same_signature <_transformers.auto_model._assert_same_signature>`
  - ```{autodoc2-docstring} _transformers.auto_model._assert_same_signature
    :summary:
    ```
* - {py:obj}`_patch_attention <_transformers.auto_model._patch_attention>`
  - ```{autodoc2-docstring} _transformers.auto_model._patch_attention
    :summary:
    ```
* - {py:obj}`_patch_liger_kernel <_transformers.auto_model._patch_liger_kernel>`
  - ```{autodoc2-docstring} _transformers.auto_model._patch_liger_kernel
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <_transformers.auto_model.logger>`
  - ```{autodoc2-docstring} _transformers.auto_model.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: _transformers.auto_model.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} _transformers.auto_model.logger
```

````

````{py:function} _assert_same_signature(original, patched)
:canonical: _transformers.auto_model._assert_same_signature

```{autodoc2-docstring} _transformers.auto_model._assert_same_signature
```
````

````{py:function} _patch_attention(obj, sdpa_method=None)
:canonical: _transformers.auto_model._patch_attention

```{autodoc2-docstring} _transformers.auto_model._patch_attention
```
````

````{py:function} _patch_liger_kernel(model)
:canonical: _transformers.auto_model._patch_liger_kernel

```{autodoc2-docstring} _transformers.auto_model._patch_liger_kernel
```
````

`````{py:class} _BaseNeMoAutoModelClass(*args, **kwargs)
:canonical: _transformers.auto_model._BaseNeMoAutoModelClass

Bases: {py:obj}`transformers.models.auto.auto_factory._BaseAutoModelClass`

```{autodoc2-docstring} _transformers.auto_model._BaseNeMoAutoModelClass
```

```{rubric} Initialization
```

```{autodoc2-docstring} _transformers.auto_model._BaseNeMoAutoModelClass.__init__
```

````{py:method} from_pretrained(pretrained_model_name_or_path, *model_args, use_liger_kernel: bool = True, use_sdpa_patching: bool = True, sdpa_method: typing.Optional[typing.List[torch.nn.attention.SDPBackend]] = None, torch_dtype='auto', attn_implementation: str = 'flash_attention_2', fp8_config: typing.Optional[object] = None, **kwargs) -> transformers.PreTrainedModel
:canonical: _transformers.auto_model._BaseNeMoAutoModelClass.from_pretrained
:classmethod:

```{autodoc2-docstring} _transformers.auto_model._BaseNeMoAutoModelClass.from_pretrained
```

````

````{py:method} from_config(config, *model_args, use_liger_kernel: bool = True, use_sdpa_patching: bool = True, sdpa_method: typing.Optional[typing.List[torch.nn.attention.SDPBackend]] = None, torch_dtype: typing.Union[str, torch.dtype] = 'auto', attn_implementation: str = 'flash_attention_2', fp8_config: typing.Optional[object] = None, **kwargs) -> transformers.PreTrainedModel
:canonical: _transformers.auto_model._BaseNeMoAutoModelClass.from_config
:classmethod:

```{autodoc2-docstring} _transformers.auto_model._BaseNeMoAutoModelClass.from_config
```

````

`````

````{py:class} NeMoAutoModelForCausalLM(*args, **kwargs)
:canonical: _transformers.auto_model.NeMoAutoModelForCausalLM

Bases: {py:obj}`_transformers.auto_model._BaseNeMoAutoModelClass`, {py:obj}`transformers.AutoModelForCausalLM`

```{autodoc2-docstring} _transformers.auto_model.NeMoAutoModelForCausalLM
```

```{rubric} Initialization
```

```{autodoc2-docstring} _transformers.auto_model.NeMoAutoModelForCausalLM.__init__
```

````

````{py:class} NeMoAutoModelForImageTextToText(*args, **kwargs)
:canonical: _transformers.auto_model.NeMoAutoModelForImageTextToText

Bases: {py:obj}`_transformers.auto_model._BaseNeMoAutoModelClass`, {py:obj}`transformers.AutoModelForImageTextToText`

```{autodoc2-docstring} _transformers.auto_model.NeMoAutoModelForImageTextToText
```

```{rubric} Initialization
```

```{autodoc2-docstring} _transformers.auto_model.NeMoAutoModelForImageTextToText.__init__
```

````
