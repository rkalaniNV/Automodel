# {py:mod}`nemo_automodel.components.distributed.init_utils`

```{py:module} nemo_automodel.components.distributed.init_utils
```

```{autodoc2-docstring} nemo_automodel.components.distributed.init_utils
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DistInfo <nemo_automodel.components.distributed.init_utils.DistInfo>`
  - ```{autodoc2-docstring} nemo_automodel.components.distributed.init_utils.DistInfo
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_rank_safe <nemo_automodel.components.distributed.init_utils.get_rank_safe>`
  - ```{autodoc2-docstring} nemo_automodel.components.distributed.init_utils.get_rank_safe
    :summary:
    ```
* - {py:obj}`get_world_size_safe <nemo_automodel.components.distributed.init_utils.get_world_size_safe>`
  - ```{autodoc2-docstring} nemo_automodel.components.distributed.init_utils.get_world_size_safe
    :summary:
    ```
* - {py:obj}`get_local_rank_preinit <nemo_automodel.components.distributed.init_utils.get_local_rank_preinit>`
  - ```{autodoc2-docstring} nemo_automodel.components.distributed.init_utils.get_local_rank_preinit
    :summary:
    ```
* - {py:obj}`initialize_distributed <nemo_automodel.components.distributed.init_utils.initialize_distributed>`
  - ```{autodoc2-docstring} nemo_automodel.components.distributed.init_utils.initialize_distributed
    :summary:
    ```
* - {py:obj}`destroy_global_state <nemo_automodel.components.distributed.init_utils.destroy_global_state>`
  - ```{autodoc2-docstring} nemo_automodel.components.distributed.init_utils.destroy_global_state
    :summary:
    ```
````

### API

````{py:function} get_rank_safe() -> int
:canonical: nemo_automodel.components.distributed.init_utils.get_rank_safe

```{autodoc2-docstring} nemo_automodel.components.distributed.init_utils.get_rank_safe
```
````

````{py:function} get_world_size_safe() -> int
:canonical: nemo_automodel.components.distributed.init_utils.get_world_size_safe

```{autodoc2-docstring} nemo_automodel.components.distributed.init_utils.get_world_size_safe
```
````

````{py:function} get_local_rank_preinit() -> int
:canonical: nemo_automodel.components.distributed.init_utils.get_local_rank_preinit

```{autodoc2-docstring} nemo_automodel.components.distributed.init_utils.get_local_rank_preinit
```
````

`````{py:class} DistInfo
:canonical: nemo_automodel.components.distributed.init_utils.DistInfo

```{autodoc2-docstring} nemo_automodel.components.distributed.init_utils.DistInfo
```

````{py:attribute} backend
:canonical: nemo_automodel.components.distributed.init_utils.DistInfo.backend
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_automodel.components.distributed.init_utils.DistInfo.backend
```

````

````{py:attribute} rank
:canonical: nemo_automodel.components.distributed.init_utils.DistInfo.rank
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_automodel.components.distributed.init_utils.DistInfo.rank
```

````

````{py:attribute} world_size
:canonical: nemo_automodel.components.distributed.init_utils.DistInfo.world_size
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_automodel.components.distributed.init_utils.DistInfo.world_size
```

````

````{py:attribute} device
:canonical: nemo_automodel.components.distributed.init_utils.DistInfo.device
:type: torch.device
:value: >
   None

```{autodoc2-docstring} nemo_automodel.components.distributed.init_utils.DistInfo.device
```

````

````{py:attribute} is_main
:canonical: nemo_automodel.components.distributed.init_utils.DistInfo.is_main
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_automodel.components.distributed.init_utils.DistInfo.is_main
```

````

`````

````{py:function} initialize_distributed(backend, timeout_minutes=1)
:canonical: nemo_automodel.components.distributed.init_utils.initialize_distributed

```{autodoc2-docstring} nemo_automodel.components.distributed.init_utils.initialize_distributed
```
````

````{py:function} destroy_global_state()
:canonical: nemo_automodel.components.distributed.init_utils.destroy_global_state

```{autodoc2-docstring} nemo_automodel.components.distributed.init_utils.destroy_global_state
```
````
