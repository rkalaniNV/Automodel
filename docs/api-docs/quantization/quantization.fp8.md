# {py:mod}`quantization.fp8`

```{py:module} quantization.fp8
```

```{autodoc2-docstring} quantization.fp8
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FP8Config <quantization.fp8.FP8Config>`
  - ```{autodoc2-docstring} quantization.fp8.FP8Config
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_has_cuda_capability <quantization.fp8._has_cuda_capability>`
  - ```{autodoc2-docstring} quantization.fp8._has_cuda_capability
    :summary:
    ```
* - {py:obj}`_module_filter_fn <quantization.fp8._module_filter_fn>`
  - ```{autodoc2-docstring} quantization.fp8._module_filter_fn
    :summary:
    ```
* - {py:obj}`apply_fp8_to_model <quantization.fp8.apply_fp8_to_model>`
  - ```{autodoc2-docstring} quantization.fp8.apply_fp8_to_model
    :summary:
    ```
* - {py:obj}`verify_fp8_conversion <quantization.fp8.verify_fp8_conversion>`
  - ```{autodoc2-docstring} quantization.fp8.verify_fp8_conversion
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <quantization.fp8.logger>`
  - ```{autodoc2-docstring} quantization.fp8.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: quantization.fp8.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} quantization.fp8.logger
```

````

`````{py:class} FP8Config
:canonical: quantization.fp8.FP8Config

```{autodoc2-docstring} quantization.fp8.FP8Config
```

````{py:attribute} recipe_name
:canonical: quantization.fp8.FP8Config.recipe_name
:type: typing.Optional[typing.Literal[tensorwise, rowwise, rowwise_with_gw_hp]]
:value: >
   None

```{autodoc2-docstring} quantization.fp8.FP8Config.recipe_name
```

````

````{py:attribute} enable_fsdp_float8_all_gather
:canonical: quantization.fp8.FP8Config.enable_fsdp_float8_all_gather
:type: bool
:value: >
   False

```{autodoc2-docstring} quantization.fp8.FP8Config.enable_fsdp_float8_all_gather
```

````

````{py:attribute} precompute_float8_dynamic_scale_for_fsdp
:canonical: quantization.fp8.FP8Config.precompute_float8_dynamic_scale_for_fsdp
:type: bool
:value: >
   False

```{autodoc2-docstring} quantization.fp8.FP8Config.precompute_float8_dynamic_scale_for_fsdp
```

````

````{py:attribute} force_recompute_fp8_weight_in_bwd
:canonical: quantization.fp8.FP8Config.force_recompute_fp8_weight_in_bwd
:type: bool
:value: >
   False

```{autodoc2-docstring} quantization.fp8.FP8Config.force_recompute_fp8_weight_in_bwd
```

````

````{py:attribute} filter_fqns
:canonical: quantization.fp8.FP8Config.filter_fqns
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} quantization.fp8.FP8Config.filter_fqns
```

````

````{py:attribute} emulate
:canonical: quantization.fp8.FP8Config.emulate
:type: bool
:value: >
   False

```{autodoc2-docstring} quantization.fp8.FP8Config.emulate
```

````

````{py:method} from_config_node(config_node)
:canonical: quantization.fp8.FP8Config.from_config_node
:classmethod:

```{autodoc2-docstring} quantization.fp8.FP8Config.from_config_node
```

````

````{py:method} to_dict()
:canonical: quantization.fp8.FP8Config.to_dict

```{autodoc2-docstring} quantization.fp8.FP8Config.to_dict
```

````

`````

````{py:function} _has_cuda_capability(major: int, minor: int) -> bool
:canonical: quantization.fp8._has_cuda_capability

```{autodoc2-docstring} quantization.fp8._has_cuda_capability
```
````

````{py:function} _module_filter_fn(module, name, filter_fqns: typing.List[str] = None)
:canonical: quantization.fp8._module_filter_fn

```{autodoc2-docstring} quantization.fp8._module_filter_fn
```
````

````{py:function} apply_fp8_to_model(model: torch.nn.Module, filter_fqns: typing.Optional[typing.List[str]] = None, recipe_name: typing.Optional[str] = None, force_recompute_fp8_weight_in_bwd: bool = False, enable_fsdp_float8_all_gather: bool = False, emulate: bool = False) -> torch.nn.Module
:canonical: quantization.fp8.apply_fp8_to_model

```{autodoc2-docstring} quantization.fp8.apply_fp8_to_model
```
````

````{py:function} verify_fp8_conversion(model: torch.nn.Module) -> dict
:canonical: quantization.fp8.verify_fp8_conversion

```{autodoc2-docstring} quantization.fp8.verify_fp8_conversion
```
````
