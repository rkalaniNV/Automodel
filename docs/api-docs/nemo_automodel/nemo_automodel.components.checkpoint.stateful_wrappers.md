# {py:mod}`nemo_automodel.components.checkpoint.stateful_wrappers`

```{py:module} nemo_automodel.components.checkpoint.stateful_wrappers
```

```{autodoc2-docstring} nemo_automodel.components.checkpoint.stateful_wrappers
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ModelState <nemo_automodel.components.checkpoint.stateful_wrappers.ModelState>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint.stateful_wrappers.ModelState
    :summary:
    ```
* - {py:obj}`OptimizerState <nemo_automodel.components.checkpoint.stateful_wrappers.OptimizerState>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint.stateful_wrappers.OptimizerState
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_drop_outer_prefix <nemo_automodel.components.checkpoint.stateful_wrappers._drop_outer_prefix>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint.stateful_wrappers._drop_outer_prefix
    :summary:
    ```
* - {py:obj}`_add_outer_prefix <nemo_automodel.components.checkpoint.stateful_wrappers._add_outer_prefix>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint.stateful_wrappers._add_outer_prefix
    :summary:
    ```
* - {py:obj}`_get_lm_head_weight_and_name <nemo_automodel.components.checkpoint.stateful_wrappers._get_lm_head_weight_and_name>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint.stateful_wrappers._get_lm_head_weight_and_name
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_PREFIX <nemo_automodel.components.checkpoint.stateful_wrappers._PREFIX>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint.stateful_wrappers._PREFIX
    :summary:
    ```
````

### API

````{py:data} _PREFIX
:canonical: nemo_automodel.components.checkpoint.stateful_wrappers._PREFIX
:value: >
   'model.'

```{autodoc2-docstring} nemo_automodel.components.checkpoint.stateful_wrappers._PREFIX
```

````

````{py:function} _drop_outer_prefix(sd: dict[str, typing.Any], prefix: str = _PREFIX) -> None
:canonical: nemo_automodel.components.checkpoint.stateful_wrappers._drop_outer_prefix

```{autodoc2-docstring} nemo_automodel.components.checkpoint.stateful_wrappers._drop_outer_prefix
```
````

````{py:function} _add_outer_prefix(sd: dict[str, typing.Any], prefix: str = _PREFIX, skip_keys: list[str] = []) -> None
:canonical: nemo_automodel.components.checkpoint.stateful_wrappers._add_outer_prefix

```{autodoc2-docstring} nemo_automodel.components.checkpoint.stateful_wrappers._add_outer_prefix
```
````

````{py:function} _get_lm_head_weight_and_name(model: torch.nn.Module) -> typing.Optional[tuple[torch.Tensor, str]]
:canonical: nemo_automodel.components.checkpoint.stateful_wrappers._get_lm_head_weight_and_name

```{autodoc2-docstring} nemo_automodel.components.checkpoint.stateful_wrappers._get_lm_head_weight_and_name
```
````

`````{py:class} ModelState(model: torch.nn.Module, is_peft: bool = False)
:canonical: nemo_automodel.components.checkpoint.stateful_wrappers.ModelState

```{autodoc2-docstring} nemo_automodel.components.checkpoint.stateful_wrappers.ModelState
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_automodel.components.checkpoint.stateful_wrappers.ModelState.__init__
```

````{py:method} state_dict() -> dict[str, typing.Any]
:canonical: nemo_automodel.components.checkpoint.stateful_wrappers.ModelState.state_dict

```{autodoc2-docstring} nemo_automodel.components.checkpoint.stateful_wrappers.ModelState.state_dict
```

````

````{py:method} load_state_dict(state_dict: dict[str, typing.Any]) -> None
:canonical: nemo_automodel.components.checkpoint.stateful_wrappers.ModelState.load_state_dict

```{autodoc2-docstring} nemo_automodel.components.checkpoint.stateful_wrappers.ModelState.load_state_dict
```

````

`````

`````{py:class} OptimizerState(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: typing.Optional[typing.Any] = None)
:canonical: nemo_automodel.components.checkpoint.stateful_wrappers.OptimizerState

```{autodoc2-docstring} nemo_automodel.components.checkpoint.stateful_wrappers.OptimizerState
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_automodel.components.checkpoint.stateful_wrappers.OptimizerState.__init__
```

````{py:method} state_dict() -> dict[str, typing.Any]
:canonical: nemo_automodel.components.checkpoint.stateful_wrappers.OptimizerState.state_dict

```{autodoc2-docstring} nemo_automodel.components.checkpoint.stateful_wrappers.OptimizerState.state_dict
```

````

````{py:method} load_state_dict(state_dict: dict[str, typing.Any]) -> None
:canonical: nemo_automodel.components.checkpoint.stateful_wrappers.OptimizerState.load_state_dict

```{autodoc2-docstring} nemo_automodel.components.checkpoint.stateful_wrappers.OptimizerState.load_state_dict
```

````

`````
