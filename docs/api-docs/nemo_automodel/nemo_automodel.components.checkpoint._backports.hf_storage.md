# {py:mod}`nemo_automodel.components.checkpoint._backports.hf_storage`

```{py:module} nemo_automodel.components.checkpoint._backports.hf_storage
```

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.hf_storage
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_HuggingFaceStorageWriter <nemo_automodel.components.checkpoint._backports.hf_storage._HuggingFaceStorageWriter>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.hf_storage._HuggingFaceStorageWriter
    :summary:
    ```
* - {py:obj}`_HuggingFaceStorageReader <nemo_automodel.components.checkpoint._backports.hf_storage._HuggingFaceStorageReader>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.hf_storage._HuggingFaceStorageReader
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_extract_file_index <nemo_automodel.components.checkpoint._backports.hf_storage._extract_file_index>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.hf_storage._extract_file_index
    :summary:
    ```
* - {py:obj}`get_fqn_to_file_index_mapping <nemo_automodel.components.checkpoint._backports.hf_storage.get_fqn_to_file_index_mapping>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.hf_storage.get_fqn_to_file_index_mapping
    :summary:
    ```
* - {py:obj}`_get_key_renaming_mapping <nemo_automodel.components.checkpoint._backports.hf_storage._get_key_renaming_mapping>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.hf_storage._get_key_renaming_mapping
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <nemo_automodel.components.checkpoint._backports.hf_storage.__all__>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.hf_storage.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: nemo_automodel.components.checkpoint._backports.hf_storage.__all__
:value: >
   ['_HuggingFaceStorageWriter', '_HuggingFaceStorageReader']

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.hf_storage.__all__
```

````

`````{py:class} _HuggingFaceStorageWriter(path: str, fqn_to_index_mapping: typing.Optional[dict[str, int]] = None, thread_count: int = 1, token: typing.Optional[str] = None, save_sharded: bool = False, consolidated_output_path: typing.Optional[str] = None, num_threads_consolidation: typing.Optional[int] = None)
:canonical: nemo_automodel.components.checkpoint._backports.hf_storage._HuggingFaceStorageWriter

Bases: {py:obj}`nemo_automodel.components.checkpoint._backports._fsspec_filesystem.FsspecWriter`

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.hf_storage._HuggingFaceStorageWriter
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.hf_storage._HuggingFaceStorageWriter.__init__
```

````{py:method} prepare_global_plan(plans: list[torch.distributed.checkpoint.planner.SavePlan]) -> list[torch.distributed.checkpoint.planner.SavePlan]
:canonical: nemo_automodel.components.checkpoint._backports.hf_storage._HuggingFaceStorageWriter.prepare_global_plan

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.hf_storage._HuggingFaceStorageWriter.prepare_global_plan
```

````

````{py:method} write_data(plan: torch.distributed.checkpoint.planner.SavePlan, planner: torch.distributed.checkpoint.planner.SavePlanner) -> torch.futures.Future[list[torch.distributed.checkpoint.storage.WriteResult]]
:canonical: nemo_automodel.components.checkpoint._backports.hf_storage._HuggingFaceStorageWriter.write_data

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.hf_storage._HuggingFaceStorageWriter.write_data
```

````

````{py:method} finish(metadata: torch.distributed.checkpoint.metadata.Metadata, results: list[list[torch.distributed.checkpoint.storage.WriteResult]]) -> None
:canonical: nemo_automodel.components.checkpoint._backports.hf_storage._HuggingFaceStorageWriter.finish

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.hf_storage._HuggingFaceStorageWriter.finish
```

````

````{py:method} _split_by_storage_plan(storage_plan: typing.Optional[dict[str, int]], items: list[torch.distributed.checkpoint.planner.WriteItem]) -> dict[int, list[torch.distributed.checkpoint.planner.WriteItem]]
:canonical: nemo_automodel.components.checkpoint._backports.hf_storage._HuggingFaceStorageWriter._split_by_storage_plan

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.hf_storage._HuggingFaceStorageWriter._split_by_storage_plan
```

````

````{py:property} metadata_path
:canonical: nemo_automodel.components.checkpoint._backports.hf_storage._HuggingFaceStorageWriter.metadata_path
:type: str

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.hf_storage._HuggingFaceStorageWriter.metadata_path
```

````

`````

`````{py:class} _HuggingFaceStorageReader(path: str, token: typing.Optional[str] = None)
:canonical: nemo_automodel.components.checkpoint._backports.hf_storage._HuggingFaceStorageReader

Bases: {py:obj}`nemo_automodel.components.checkpoint._backports._fsspec_filesystem.FsspecReader`

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.hf_storage._HuggingFaceStorageReader
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.hf_storage._HuggingFaceStorageReader.__init__
```

````{py:method} read_data(plan: torch.distributed.checkpoint.planner.LoadPlan, planner: torch.distributed.checkpoint.planner.LoadPlanner) -> torch.futures.Future[None]
:canonical: nemo_automodel.components.checkpoint._backports.hf_storage._HuggingFaceStorageReader.read_data

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.hf_storage._HuggingFaceStorageReader.read_data
```

````

````{py:method} read_metadata() -> torch.distributed.checkpoint.metadata.Metadata
:canonical: nemo_automodel.components.checkpoint._backports.hf_storage._HuggingFaceStorageReader.read_metadata

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.hf_storage._HuggingFaceStorageReader.read_metadata
```

````

`````

````{py:function} _extract_file_index(filename: str) -> int
:canonical: nemo_automodel.components.checkpoint._backports.hf_storage._extract_file_index

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.hf_storage._extract_file_index
```
````

````{py:function} get_fqn_to_file_index_mapping(reference_model_path: str, key_mapping: typing.Optional[dict[str, str]] = None) -> dict[str, int]
:canonical: nemo_automodel.components.checkpoint._backports.hf_storage.get_fqn_to_file_index_mapping

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.hf_storage.get_fqn_to_file_index_mapping
```
````

````{py:function} _get_key_renaming_mapping(key: str, key_mapping: typing.Optional[dict[str, str]] = None) -> str
:canonical: nemo_automodel.components.checkpoint._backports.hf_storage._get_key_renaming_mapping

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.hf_storage._get_key_renaming_mapping
```
````
