# {py:mod}`distributed.tensor_utils`

```{py:module} distributed.tensor_utils
```

```{autodoc2-docstring} distributed.tensor_utils
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_cpu_state_dict <distributed.tensor_utils.get_cpu_state_dict>`
  - ```{autodoc2-docstring} distributed.tensor_utils.get_cpu_state_dict
    :summary:
    ```
* - {py:obj}`to_cpu <distributed.tensor_utils.to_cpu>`
  - ```{autodoc2-docstring} distributed.tensor_utils.to_cpu
    :summary:
    ```
* - {py:obj}`to_local_if_dtensor <distributed.tensor_utils.to_local_if_dtensor>`
  - ```{autodoc2-docstring} distributed.tensor_utils.to_local_if_dtensor
    :summary:
    ```
````

### API

````{py:function} get_cpu_state_dict(state_generator: typing.Iterable[tuple[str, typing.Union[torch.Tensor, torch.distributed.tensor.DTensor]]], pin_memory: bool = False) -> dict[str, torch.Tensor]
:canonical: distributed.tensor_utils.get_cpu_state_dict

```{autodoc2-docstring} distributed.tensor_utils.get_cpu_state_dict
```
````

````{py:function} to_cpu(v)
:canonical: distributed.tensor_utils.to_cpu

```{autodoc2-docstring} distributed.tensor_utils.to_cpu
```
````

````{py:function} to_local_if_dtensor(tensor: typing.Union[torch.Tensor, torch.distributed.tensor.DTensor]) -> torch.Tensor
:canonical: distributed.tensor_utils.to_local_if_dtensor

```{autodoc2-docstring} distributed.tensor_utils.to_local_if_dtensor
```
````
