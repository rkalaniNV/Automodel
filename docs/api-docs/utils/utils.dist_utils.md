# {py:mod}`utils.dist_utils`

```{py:module} utils.dist_utils
```

```{autodoc2-docstring} utils.dist_utils
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FirstRankPerNode <utils.dist_utils.FirstRankPerNode>`
  - ```{autodoc2-docstring} utils.dist_utils.FirstRankPerNode
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`barrier_and_log <utils.dist_utils.barrier_and_log>`
  - ```{autodoc2-docstring} utils.dist_utils.barrier_and_log
    :summary:
    ```
* - {py:obj}`reduce_loss <utils.dist_utils.reduce_loss>`
  - ```{autodoc2-docstring} utils.dist_utils.reduce_loss
    :summary:
    ```
* - {py:obj}`get_sync_ctx <utils.dist_utils.get_sync_ctx>`
  - ```{autodoc2-docstring} utils.dist_utils.get_sync_ctx
    :summary:
    ```
* - {py:obj}`rescale_gradients <utils.dist_utils.rescale_gradients>`
  - ```{autodoc2-docstring} utils.dist_utils.rescale_gradients
    :summary:
    ```
* - {py:obj}`clip_gradients <utils.dist_utils.clip_gradients>`
  - ```{autodoc2-docstring} utils.dist_utils.clip_gradients
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <utils.dist_utils.logger>`
  - ```{autodoc2-docstring} utils.dist_utils.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: utils.dist_utils.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} utils.dist_utils.logger
```

````

`````{py:class} FirstRankPerNode
:canonical: utils.dist_utils.FirstRankPerNode

Bases: {py:obj}`contextlib.ContextDecorator`

```{autodoc2-docstring} utils.dist_utils.FirstRankPerNode
```

````{py:method} __enter__()
:canonical: utils.dist_utils.FirstRankPerNode.__enter__

```{autodoc2-docstring} utils.dist_utils.FirstRankPerNode.__enter__
```

````

````{py:method} __exit__(exc_type, exc_val, exc_tb)
:canonical: utils.dist_utils.FirstRankPerNode.__exit__

```{autodoc2-docstring} utils.dist_utils.FirstRankPerNode.__exit__
```

````

````{py:method} _try_bootstrap_pg() -> bool
:canonical: utils.dist_utils.FirstRankPerNode._try_bootstrap_pg

```{autodoc2-docstring} utils.dist_utils.FirstRankPerNode._try_bootstrap_pg
```

````

`````

````{py:function} barrier_and_log(string: str) -> None
:canonical: utils.dist_utils.barrier_and_log

```{autodoc2-docstring} utils.dist_utils.barrier_and_log
```
````

````{py:function} reduce_loss(loss_store: list[torch.Tensor], total_num_tokens: torch.Tensor, per_token_loss: bool = True, dp_group: typing.Optional[torch.distributed.ProcessGroup] = None) -> tuple[torch.Tensor, torch.Tensor]
:canonical: utils.dist_utils.reduce_loss

```{autodoc2-docstring} utils.dist_utils.reduce_loss
```
````

````{py:function} get_sync_ctx(model, is_optim_step)
:canonical: utils.dist_utils.get_sync_ctx

```{autodoc2-docstring} utils.dist_utils.get_sync_ctx
```
````

````{py:function} rescale_gradients(model, num_tokens_for_grad_scaling, dp_group=None)
:canonical: utils.dist_utils.rescale_gradients

```{autodoc2-docstring} utils.dist_utils.rescale_gradients
```
````

````{py:function} clip_gradients(model, clip_norm, foreach=True)
:canonical: utils.dist_utils.clip_gradients

```{autodoc2-docstring} utils.dist_utils.clip_gradients
```
````
