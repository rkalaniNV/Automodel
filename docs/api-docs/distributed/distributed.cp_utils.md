# {py:mod}`distributed.cp_utils`

```{py:module} distributed.cp_utils
```

```{autodoc2-docstring} distributed.cp_utils
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_build_position_ids <distributed.cp_utils._build_position_ids>`
  - ```{autodoc2-docstring} distributed.cp_utils._build_position_ids
    :summary:
    ```
* - {py:obj}`get_train_context <distributed.cp_utils.get_train_context>`
  - ```{autodoc2-docstring} distributed.cp_utils.get_train_context
    :summary:
    ```
* - {py:obj}`create_context_parallel_ctx <distributed.cp_utils.create_context_parallel_ctx>`
  - ```{autodoc2-docstring} distributed.cp_utils.create_context_parallel_ctx
    :summary:
    ```
* - {py:obj}`make_cp_batch_and_ctx <distributed.cp_utils.make_cp_batch_and_ctx>`
  - ```{autodoc2-docstring} distributed.cp_utils.make_cp_batch_and_ctx
    :summary:
    ```
````

### API

````{py:function} _build_position_ids(batch, device)
:canonical: distributed.cp_utils._build_position_ids

```{autodoc2-docstring} distributed.cp_utils._build_position_ids
```
````

````{py:function} get_train_context(enable_loss_parallel: bool, enable_compiled_autograd: bool, cp_context=None)
:canonical: distributed.cp_utils.get_train_context

```{autodoc2-docstring} distributed.cp_utils.get_train_context
```
````

````{py:function} create_context_parallel_ctx(cp_mesh: torch.distributed.device_mesh.DeviceMesh, cp_buffers: typing.List[torch.Tensor], cp_seq_dims: typing.List[int], cp_no_restore_buffers: typing.Set[torch.Tensor], cp_rotate_method: typing.Optional[str] = None)
:canonical: distributed.cp_utils.create_context_parallel_ctx

```{autodoc2-docstring} distributed.cp_utils.create_context_parallel_ctx
```
````

````{py:function} make_cp_batch_and_ctx(device_mesh, batch, labels, loss_mask)
:canonical: distributed.cp_utils.make_cp_batch_and_ctx

```{autodoc2-docstring} distributed.cp_utils.make_cp_batch_and_ctx
```
````
