# {py:mod}`utils.model_utils`

```{py:module} utils.model_utils
```

```{autodoc2-docstring} utils.model_utils
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_get_model_param_stats <utils.model_utils._get_model_param_stats>`
  - ```{autodoc2-docstring} utils.model_utils._get_model_param_stats
    :summary:
    ```
* - {py:obj}`print_trainable_parameters <utils.model_utils.print_trainable_parameters>`
  - ```{autodoc2-docstring} utils.model_utils.print_trainable_parameters
    :summary:
    ```
* - {py:obj}`_freeze_module_by_attribute_and_patterns <utils.model_utils._freeze_module_by_attribute_and_patterns>`
  - ```{autodoc2-docstring} utils.model_utils._freeze_module_by_attribute_and_patterns
    :summary:
    ```
* - {py:obj}`apply_parameter_freezing <utils.model_utils.apply_parameter_freezing>`
  - ```{autodoc2-docstring} utils.model_utils.apply_parameter_freezing
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <utils.model_utils.logger>`
  - ```{autodoc2-docstring} utils.model_utils.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: utils.model_utils.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} utils.model_utils.logger
```

````

````{py:function} _get_model_param_stats(model: torch.nn.Module) -> tuple[int, int, float]
:canonical: utils.model_utils._get_model_param_stats

```{autodoc2-docstring} utils.model_utils._get_model_param_stats
```
````

````{py:function} print_trainable_parameters(model: torch.nn.Module) -> tuple[int, int]
:canonical: utils.model_utils.print_trainable_parameters

```{autodoc2-docstring} utils.model_utils.print_trainable_parameters
```
````

````{py:function} _freeze_module_by_attribute_and_patterns(model, attribute_name, name_patterns)
:canonical: utils.model_utils._freeze_module_by_attribute_and_patterns

```{autodoc2-docstring} utils.model_utils._freeze_module_by_attribute_and_patterns
```
````

````{py:function} apply_parameter_freezing(model, freeze_config)
:canonical: utils.model_utils.apply_parameter_freezing

```{autodoc2-docstring} utils.model_utils.apply_parameter_freezing
```
````
