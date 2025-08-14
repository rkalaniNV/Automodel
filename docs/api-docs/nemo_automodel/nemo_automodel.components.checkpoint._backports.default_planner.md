# {py:mod}`nemo_automodel.components.checkpoint._backports.default_planner`

```{py:module} nemo_automodel.components.checkpoint._backports.default_planner
```

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DefaultSavePlanner <nemo_automodel.components.checkpoint._backports.default_planner.DefaultSavePlanner>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.DefaultSavePlanner
    :summary:
    ```
* - {py:obj}`DefaultLoadPlanner <nemo_automodel.components.checkpoint._backports.default_planner.DefaultLoadPlanner>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.DefaultLoadPlanner
    :summary:
    ```
* - {py:obj}`_EmptyStateDictLoadPlanner <nemo_automodel.components.checkpoint._backports.default_planner._EmptyStateDictLoadPlanner>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner._EmptyStateDictLoadPlanner
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_default_local_load_plan <nemo_automodel.components.checkpoint._backports.default_planner.create_default_local_load_plan>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.create_default_local_load_plan
    :summary:
    ```
* - {py:obj}`create_default_global_load_plan <nemo_automodel.components.checkpoint._backports.default_planner.create_default_global_load_plan>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.create_default_global_load_plan
    :summary:
    ```
* - {py:obj}`create_default_local_save_plan <nemo_automodel.components.checkpoint._backports.default_planner.create_default_local_save_plan>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.create_default_local_save_plan
    :summary:
    ```
* - {py:obj}`create_default_global_save_plan <nemo_automodel.components.checkpoint._backports.default_planner.create_default_global_save_plan>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.create_default_global_save_plan
    :summary:
    ```
* - {py:obj}`_create_default_local_metadata <nemo_automodel.components.checkpoint._backports.default_planner._create_default_local_metadata>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner._create_default_local_metadata
    :summary:
    ```
* - {py:obj}`_check_box_overlap <nemo_automodel.components.checkpoint._backports.default_planner._check_box_overlap>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner._check_box_overlap
    :summary:
    ```
* - {py:obj}`_check_box_bounds <nemo_automodel.components.checkpoint._backports.default_planner._check_box_bounds>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner._check_box_bounds
    :summary:
    ```
* - {py:obj}`_validate_global_plan <nemo_automodel.components.checkpoint._backports.default_planner._validate_global_plan>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner._validate_global_plan
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <nemo_automodel.components.checkpoint._backports.default_planner.logger>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.logger
    :summary:
    ```
* - {py:obj}`__all__ <nemo_automodel.components.checkpoint._backports.default_planner.__all__>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.__all__
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.logger
:type: logging.Logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.logger
```

````

````{py:data} __all__
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.__all__
:value: >
   ['DefaultSavePlanner', 'DefaultLoadPlanner', 'create_default_local_load_plan', 'create_default_globa...

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.__all__
```

````

`````{py:class} DefaultSavePlanner(flatten_state_dict: bool = True, flatten_sharded_tensors: bool = True, dedup_replicated_tensors: typing.Optional[bool] = None, dedup_save_to_lowest_rank: bool = False, enable_plan_caching: bool = False)
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.DefaultSavePlanner

Bases: {py:obj}`torch.distributed.checkpoint.planner.SavePlanner`

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.DefaultSavePlanner
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.DefaultSavePlanner.__init__
```

````{py:attribute} mappings
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.DefaultSavePlanner.mappings
:type: torch.distributed.checkpoint._nested_dict.FLATTEN_MAPPING
:value: >
   None

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.DefaultSavePlanner.mappings
```

````

````{py:method} set_up_planner(state_dict: torch.distributed.checkpoint.metadata.STATE_DICT_TYPE, storage_meta: typing.Optional[torch.distributed.checkpoint.metadata.StorageMeta] = None, is_coordinator: bool = False) -> None
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.DefaultSavePlanner.set_up_planner

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.DefaultSavePlanner.set_up_planner
```

````

````{py:method} create_local_plan() -> torch.distributed.checkpoint.planner.SavePlan
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.DefaultSavePlanner.create_local_plan

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.DefaultSavePlanner.create_local_plan
```

````

````{py:method} _dedup_save_plans(all_plans: list[torch.distributed.checkpoint.planner.SavePlan]) -> list[torch.distributed.checkpoint.planner.SavePlan]
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.DefaultSavePlanner._dedup_save_plans

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.DefaultSavePlanner._dedup_save_plans
```

````

````{py:method} _create_global_plan(all_plans: list[torch.distributed.checkpoint.planner.SavePlan]) -> tuple[list[torch.distributed.checkpoint.planner.SavePlan], torch.distributed.checkpoint.metadata.Metadata]
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.DefaultSavePlanner._create_global_plan

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.DefaultSavePlanner._create_global_plan
```

````

````{py:method} _create_global_plan_with_caching(all_plans: list[torch.distributed.checkpoint.planner.SavePlan]) -> tuple[list[torch.distributed.checkpoint.planner.SavePlan], list[torch.distributed.checkpoint.planner.SavePlan], torch.distributed.checkpoint.metadata.Metadata]
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.DefaultSavePlanner._create_global_plan_with_caching

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.DefaultSavePlanner._create_global_plan_with_caching
```

````

````{py:method} create_global_plan(all_plans: list[torch.distributed.checkpoint.planner.SavePlan]) -> tuple[list[torch.distributed.checkpoint.planner.SavePlan], torch.distributed.checkpoint.metadata.Metadata]
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.DefaultSavePlanner.create_global_plan

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.DefaultSavePlanner.create_global_plan
```

````

````{py:method} _finish_plan_with_caching(new_plan: torch.distributed.checkpoint.planner.SavePlan) -> torch.distributed.checkpoint.planner.SavePlan
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.DefaultSavePlanner._finish_plan_with_caching

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.DefaultSavePlanner._finish_plan_with_caching
```

````

````{py:method} finish_plan(new_plan: torch.distributed.checkpoint.planner.SavePlan) -> torch.distributed.checkpoint.planner.SavePlan
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.DefaultSavePlanner.finish_plan

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.DefaultSavePlanner.finish_plan
```

````

````{py:method} resolve_data(write_item: torch.distributed.checkpoint.planner.WriteItem) -> typing.Union[torch.Tensor, io.BytesIO]
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.DefaultSavePlanner.resolve_data

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.DefaultSavePlanner.resolve_data
```

````

````{py:method} lookup_object(index: torch.distributed.checkpoint.metadata.MetadataIndex) -> typing.Any
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.DefaultSavePlanner.lookup_object

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.DefaultSavePlanner.lookup_object
```

````

````{py:method} transform_object(write_item: torch.distributed.checkpoint.planner.WriteItem, object: typing.Any)
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.DefaultSavePlanner.transform_object

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.DefaultSavePlanner.transform_object
```

````

`````

`````{py:class} DefaultLoadPlanner(flatten_state_dict: bool = True, flatten_sharded_tensors: bool = True, allow_partial_load: bool = False)
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.DefaultLoadPlanner

Bases: {py:obj}`torch.distributed.checkpoint.planner.LoadPlanner`

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.DefaultLoadPlanner
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.DefaultLoadPlanner.__init__
```

````{py:attribute} original_state_dict
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.DefaultLoadPlanner.original_state_dict
:type: torch.distributed.checkpoint.metadata.STATE_DICT_TYPE
:value: >
   None

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.DefaultLoadPlanner.original_state_dict
```

````

````{py:attribute} mappings
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.DefaultLoadPlanner.mappings
:type: torch.distributed.checkpoint._nested_dict.FLATTEN_MAPPING
:value: >
   None

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.DefaultLoadPlanner.mappings
```

````

````{py:method} set_up_planner(state_dict: torch.distributed.checkpoint.metadata.STATE_DICT_TYPE, metadata: typing.Optional[torch.distributed.checkpoint.metadata.Metadata] = None, is_coordinator: bool = False) -> None
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.DefaultLoadPlanner.set_up_planner

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.DefaultLoadPlanner.set_up_planner
```

````

````{py:method} create_local_plan() -> torch.distributed.checkpoint.planner.LoadPlan
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.DefaultLoadPlanner.create_local_plan

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.DefaultLoadPlanner.create_local_plan
```

````

````{py:method} create_global_plan(global_plan: list[torch.distributed.checkpoint.planner.LoadPlan]) -> list[torch.distributed.checkpoint.planner.LoadPlan]
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.DefaultLoadPlanner.create_global_plan

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.DefaultLoadPlanner.create_global_plan
```

````

````{py:method} finish_plan(new_plan: torch.distributed.checkpoint.planner.LoadPlan) -> torch.distributed.checkpoint.planner.LoadPlan
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.DefaultLoadPlanner.finish_plan

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.DefaultLoadPlanner.finish_plan
```

````

````{py:method} load_bytes(read_item: torch.distributed.checkpoint.planner.ReadItem, value: io.BytesIO) -> None
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.DefaultLoadPlanner.load_bytes

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.DefaultLoadPlanner.load_bytes
```

````

````{py:method} resolve_tensor(read_item: torch.distributed.checkpoint.planner.ReadItem)
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.DefaultLoadPlanner.resolve_tensor

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.DefaultLoadPlanner.resolve_tensor
```

````

````{py:method} commit_tensor(read_item: torch.distributed.checkpoint.planner.ReadItem, tensor: torch.Tensor) -> None
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.DefaultLoadPlanner.commit_tensor

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.DefaultLoadPlanner.commit_tensor
```

````

````{py:method} lookup_tensor(index: torch.distributed.checkpoint.metadata.MetadataIndex) -> torch.Tensor
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.DefaultLoadPlanner.lookup_tensor

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.DefaultLoadPlanner.lookup_tensor
```

````

````{py:method} transform_tensor(read_item: torch.distributed.checkpoint.planner.ReadItem, tensor: torch.Tensor)
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.DefaultLoadPlanner.transform_tensor

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.DefaultLoadPlanner.transform_tensor
```

````

`````

`````{py:class} _EmptyStateDictLoadPlanner(keys=None, *args, **kwargs)
:canonical: nemo_automodel.components.checkpoint._backports.default_planner._EmptyStateDictLoadPlanner

Bases: {py:obj}`nemo_automodel.components.checkpoint._backports.default_planner.DefaultLoadPlanner`

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner._EmptyStateDictLoadPlanner
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner._EmptyStateDictLoadPlanner.__init__
```

````{py:method} _should_include_key(key: str, metadata: torch.distributed.checkpoint.metadata.Metadata) -> bool
:canonical: nemo_automodel.components.checkpoint._backports.default_planner._EmptyStateDictLoadPlanner._should_include_key

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner._EmptyStateDictLoadPlanner._should_include_key
```

````

````{py:method} set_up_planner(state_dict: torch.distributed.checkpoint.metadata.STATE_DICT_TYPE, metadata: typing.Optional[torch.distributed.checkpoint.metadata.Metadata] = None, is_coordinator: bool = False) -> None
:canonical: nemo_automodel.components.checkpoint._backports.default_planner._EmptyStateDictLoadPlanner.set_up_planner

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner._EmptyStateDictLoadPlanner.set_up_planner
```

````

`````

````{py:function} create_default_local_load_plan(state_dict: dict[str, typing.Any], metadata: torch.distributed.checkpoint.metadata.Metadata, strict: bool = True) -> torch.distributed.checkpoint.planner.LoadPlan
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.create_default_local_load_plan

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.create_default_local_load_plan
```
````

````{py:function} create_default_global_load_plan(all_plans: list[torch.distributed.checkpoint.planner.LoadPlan]) -> list[torch.distributed.checkpoint.planner.LoadPlan]
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.create_default_global_load_plan

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.create_default_global_load_plan
```
````

````{py:function} create_default_local_save_plan(state_dict: dict[str, typing.Any], is_coordinator: bool) -> torch.distributed.checkpoint.planner.SavePlan
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.create_default_local_save_plan

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.create_default_local_save_plan
```
````

````{py:function} create_default_global_save_plan(all_plans: list[torch.distributed.checkpoint.planner.SavePlan], rewrite_index_hints: bool = True) -> tuple[list[torch.distributed.checkpoint.planner.SavePlan], torch.distributed.checkpoint.metadata.Metadata]
:canonical: nemo_automodel.components.checkpoint._backports.default_planner.create_default_global_save_plan

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner.create_default_global_save_plan
```
````

````{py:function} _create_default_local_metadata(state_dict: torch.distributed.checkpoint.metadata.STATE_DICT_TYPE) -> torch.distributed.checkpoint.metadata.Metadata
:canonical: nemo_automodel.components.checkpoint._backports.default_planner._create_default_local_metadata

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner._create_default_local_metadata
```
````

````{py:function} _check_box_overlap(box0: torch.distributed.checkpoint.metadata.ChunkStorageMetadata, box1: torch.distributed.checkpoint.metadata.ChunkStorageMetadata) -> bool
:canonical: nemo_automodel.components.checkpoint._backports.default_planner._check_box_overlap

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner._check_box_overlap
```
````

````{py:function} _check_box_bounds(outer_box_size: torch.Size, inner_box: torch.distributed.checkpoint.metadata.ChunkStorageMetadata) -> bool
:canonical: nemo_automodel.components.checkpoint._backports.default_planner._check_box_bounds

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner._check_box_bounds
```
````

````{py:function} _validate_global_plan(global_plan: list[torch.distributed.checkpoint.planner.SavePlan], metadata: torch.distributed.checkpoint.metadata.Metadata) -> bool
:canonical: nemo_automodel.components.checkpoint._backports.default_planner._validate_global_plan

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.default_planner._validate_global_plan
```
````
