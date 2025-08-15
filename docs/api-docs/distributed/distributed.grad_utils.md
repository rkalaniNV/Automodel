# {py:mod}`distributed.grad_utils`

```{py:module} distributed.grad_utils
```

```{autodoc2-docstring} distributed.grad_utils
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`clip_grad_by_total_norm_ <distributed.grad_utils.clip_grad_by_total_norm_>`
  - ```{autodoc2-docstring} distributed.grad_utils.clip_grad_by_total_norm_
    :summary:
    ```
* - {py:obj}`get_grad_norm <distributed.grad_utils.get_grad_norm>`
  - ```{autodoc2-docstring} distributed.grad_utils.get_grad_norm
    :summary:
    ```
````

### API

````{py:function} clip_grad_by_total_norm_(parameters: typing.Union[list[typing.Union[torch.Tensor, torch.distributed.tensor.DTensor]], typing.Union[torch.Tensor, torch.distributed.tensor.DTensor]], max_grad_norm: typing.Union[int, float], total_norm: float, dtype: torch.dtype = torch.float32)
:canonical: distributed.grad_utils.clip_grad_by_total_norm_

```{autodoc2-docstring} distributed.grad_utils.clip_grad_by_total_norm_
```
````

````{py:function} get_grad_norm(parameters: typing.Union[list[typing.Union[torch.Tensor, torch.distributed.tensor.DTensor]], typing.Union[torch.Tensor, torch.distributed.tensor.DTensor]], dp_cp_group: torch.distributed.ProcessGroup, tp_group: torch.distributed.ProcessGroup, norm_type: typing.Union[int, float] = 2, dtype: torch.dtype = torch.float32) -> float
:canonical: distributed.grad_utils.get_grad_norm

```{autodoc2-docstring} distributed.grad_utils.get_grad_norm
```
````
