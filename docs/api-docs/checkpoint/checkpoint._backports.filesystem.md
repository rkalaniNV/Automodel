# {py:mod}`checkpoint._backports.filesystem`

```{py:module} checkpoint._backports.filesystem
```

```{autodoc2-docstring} checkpoint._backports.filesystem
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SerializationFormat <checkpoint._backports.filesystem.SerializationFormat>`
  - ```{autodoc2-docstring} checkpoint._backports.filesystem.SerializationFormat
    :summary:
    ```
* - {py:obj}`_StorageInfo <checkpoint._backports.filesystem._StorageInfo>`
  - ```{autodoc2-docstring} checkpoint._backports.filesystem._StorageInfo
    :summary:
    ```
* - {py:obj}`_StoragePrefix <checkpoint._backports.filesystem._StoragePrefix>`
  - ```{autodoc2-docstring} checkpoint._backports.filesystem._StoragePrefix
    :summary:
    ```
* - {py:obj}`_TensorLoader <checkpoint._backports.filesystem._TensorLoader>`
  -
* - {py:obj}`_SerialCpuLoader <checkpoint._backports.filesystem._SerialCpuLoader>`
  -
* - {py:obj}`_OverlappingCpuLoader <checkpoint._backports.filesystem._OverlappingCpuLoader>`
  -
* - {py:obj}`_StorageWriterTransforms <checkpoint._backports.filesystem._StorageWriterTransforms>`
  - ```{autodoc2-docstring} checkpoint._backports.filesystem._StorageWriterTransforms
    :summary:
    ```
* - {py:obj}`FileSystemBase <checkpoint._backports.filesystem.FileSystemBase>`
  -
* - {py:obj}`FileSystem <checkpoint._backports.filesystem.FileSystem>`
  -
* - {py:obj}`_FileSystemWriter <checkpoint._backports.filesystem._FileSystemWriter>`
  - ```{autodoc2-docstring} checkpoint._backports.filesystem._FileSystemWriter
    :summary:
    ```
* - {py:obj}`_StorageReaderTransforms <checkpoint._backports.filesystem._StorageReaderTransforms>`
  - ```{autodoc2-docstring} checkpoint._backports.filesystem._StorageReaderTransforms
    :summary:
    ```
* - {py:obj}`FileSystemReader <checkpoint._backports.filesystem.FileSystemReader>`
  -
* - {py:obj}`FileSystemWriter <checkpoint._backports.filesystem.FileSystemWriter>`
  - ```{autodoc2-docstring} checkpoint._backports.filesystem.FileSystemWriter
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_generate_uuid <checkpoint._backports.filesystem._generate_uuid>`
  - ```{autodoc2-docstring} checkpoint._backports.filesystem._generate_uuid
    :summary:
    ```
* - {py:obj}`_item_size <checkpoint._backports.filesystem._item_size>`
  - ```{autodoc2-docstring} checkpoint._backports.filesystem._item_size
    :summary:
    ```
* - {py:obj}`_split_by_size_and_type <checkpoint._backports.filesystem._split_by_size_and_type>`
  - ```{autodoc2-docstring} checkpoint._backports.filesystem._split_by_size_and_type
    :summary:
    ```
* - {py:obj}`_write_item <checkpoint._backports.filesystem._write_item>`
  - ```{autodoc2-docstring} checkpoint._backports.filesystem._write_item
    :summary:
    ```
* - {py:obj}`_write_files_from_queue <checkpoint._backports.filesystem._write_files_from_queue>`
  - ```{autodoc2-docstring} checkpoint._backports.filesystem._write_files_from_queue
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <checkpoint._backports.filesystem.__all__>`
  - ```{autodoc2-docstring} checkpoint._backports.filesystem.__all__
    :summary:
    ```
* - {py:obj}`_metadata_fn <checkpoint._backports.filesystem._metadata_fn>`
  - ```{autodoc2-docstring} checkpoint._backports.filesystem._metadata_fn
    :summary:
    ```
* - {py:obj}`DEFAULT_SUFFIX <checkpoint._backports.filesystem.DEFAULT_SUFFIX>`
  - ```{autodoc2-docstring} checkpoint._backports.filesystem.DEFAULT_SUFFIX
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: checkpoint._backports.filesystem.__all__
:value: >
   ['FileSystemWriter', 'FileSystemReader', 'FileSystem', 'FileSystemBase']

```{autodoc2-docstring} checkpoint._backports.filesystem.__all__
```

````

````{py:data} _metadata_fn
:canonical: checkpoint._backports.filesystem._metadata_fn
:type: str
:value: >
   '.metadata'

```{autodoc2-docstring} checkpoint._backports.filesystem._metadata_fn
```

````

`````{py:class} SerializationFormat(*args, **kwds)
:canonical: checkpoint._backports.filesystem.SerializationFormat

Bases: {py:obj}`enum.Enum`

```{autodoc2-docstring} checkpoint._backports.filesystem.SerializationFormat
```

```{rubric} Initialization
```

```{autodoc2-docstring} checkpoint._backports.filesystem.SerializationFormat.__init__
```

````{py:attribute} TORCH_SAVE
:canonical: checkpoint._backports.filesystem.SerializationFormat.TORCH_SAVE
:value: >
   'torch_save'

```{autodoc2-docstring} checkpoint._backports.filesystem.SerializationFormat.TORCH_SAVE
```

````

````{py:attribute} SAFETENSORS
:canonical: checkpoint._backports.filesystem.SerializationFormat.SAFETENSORS
:value: >
   'safetensors'

```{autodoc2-docstring} checkpoint._backports.filesystem.SerializationFormat.SAFETENSORS
```

````

`````

`````{py:class} _StorageInfo
:canonical: checkpoint._backports.filesystem._StorageInfo

```{autodoc2-docstring} checkpoint._backports.filesystem._StorageInfo
```

````{py:attribute} relative_path
:canonical: checkpoint._backports.filesystem._StorageInfo.relative_path
:type: str
:value: >
   None

```{autodoc2-docstring} checkpoint._backports.filesystem._StorageInfo.relative_path
```

````

````{py:attribute} offset
:canonical: checkpoint._backports.filesystem._StorageInfo.offset
:type: int
:value: >
   None

```{autodoc2-docstring} checkpoint._backports.filesystem._StorageInfo.offset
```

````

````{py:attribute} length
:canonical: checkpoint._backports.filesystem._StorageInfo.length
:type: int
:value: >
   None

```{autodoc2-docstring} checkpoint._backports.filesystem._StorageInfo.length
```

````

````{py:attribute} transform_descriptors
:canonical: checkpoint._backports.filesystem._StorageInfo.transform_descriptors
:type: typing.Optional[collections.abc.Sequence[str]]
:value: >
   None

```{autodoc2-docstring} checkpoint._backports.filesystem._StorageInfo.transform_descriptors
```

````

````{py:method} __getstate__()
:canonical: checkpoint._backports.filesystem._StorageInfo.__getstate__

````

`````

`````{py:class} _StoragePrefix
:canonical: checkpoint._backports.filesystem._StoragePrefix

```{autodoc2-docstring} checkpoint._backports.filesystem._StoragePrefix
```

````{py:attribute} prefix
:canonical: checkpoint._backports.filesystem._StoragePrefix.prefix
:type: str
:value: >
   None

```{autodoc2-docstring} checkpoint._backports.filesystem._StoragePrefix.prefix
```

````

`````

````{py:data} DEFAULT_SUFFIX
:canonical: checkpoint._backports.filesystem.DEFAULT_SUFFIX
:value: >
   '.distcp'

```{autodoc2-docstring} checkpoint._backports.filesystem.DEFAULT_SUFFIX
```

````

````{py:function} _generate_uuid() -> str
:canonical: checkpoint._backports.filesystem._generate_uuid

```{autodoc2-docstring} checkpoint._backports.filesystem._generate_uuid
```
````

`````{py:class} _TensorLoader
:canonical: checkpoint._backports.filesystem._TensorLoader

Bases: {py:obj}`abc.ABC`

````{py:method} add(size: int, obj: object) -> None
:canonical: checkpoint._backports.filesystem._TensorLoader.add
:abstractmethod:

```{autodoc2-docstring} checkpoint._backports.filesystem._TensorLoader.add
```

````

````{py:method} start_loading() -> None
:canonical: checkpoint._backports.filesystem._TensorLoader.start_loading
:abstractmethod:

```{autodoc2-docstring} checkpoint._backports.filesystem._TensorLoader.start_loading
```

````

````{py:method} values() -> collections.abc.Iterator[tuple[torch.Tensor, object]]
:canonical: checkpoint._backports.filesystem._TensorLoader.values
:abstractmethod:

```{autodoc2-docstring} checkpoint._backports.filesystem._TensorLoader.values
```

````

`````

`````{py:class} _SerialCpuLoader(resolve_fun: typing.Callable)
:canonical: checkpoint._backports.filesystem._SerialCpuLoader

Bases: {py:obj}`checkpoint._backports.filesystem._TensorLoader`

````{py:method} add(size: int, obj: object) -> None
:canonical: checkpoint._backports.filesystem._SerialCpuLoader.add

```{autodoc2-docstring} checkpoint._backports.filesystem._SerialCpuLoader.add
```

````

````{py:method} start_loading() -> None
:canonical: checkpoint._backports.filesystem._SerialCpuLoader.start_loading

```{autodoc2-docstring} checkpoint._backports.filesystem._SerialCpuLoader.start_loading
```

````

````{py:method} values() -> collections.abc.Iterator[tuple[torch.Tensor, object]]
:canonical: checkpoint._backports.filesystem._SerialCpuLoader.values

```{autodoc2-docstring} checkpoint._backports.filesystem._SerialCpuLoader.values
```

````

`````

`````{py:class} _OverlappingCpuLoader(resolve_fun: typing.Callable, stream: typing.Optional[torch.Stream] = None, inflight_threshhold: int = 1000000)
:canonical: checkpoint._backports.filesystem._OverlappingCpuLoader

Bases: {py:obj}`checkpoint._backports.filesystem._TensorLoader`

````{py:property} _done
:canonical: checkpoint._backports.filesystem._OverlappingCpuLoader._done
:type: bool

```{autodoc2-docstring} checkpoint._backports.filesystem._OverlappingCpuLoader._done
```

````

````{py:method} _drain() -> list[tuple[torch.Tensor, object]]
:canonical: checkpoint._backports.filesystem._OverlappingCpuLoader._drain

```{autodoc2-docstring} checkpoint._backports.filesystem._OverlappingCpuLoader._drain
```

````

````{py:method} _refill() -> None
:canonical: checkpoint._backports.filesystem._OverlappingCpuLoader._refill

```{autodoc2-docstring} checkpoint._backports.filesystem._OverlappingCpuLoader._refill
```

````

````{py:method} _finish() -> collections.abc.Iterable[tuple[torch.Tensor, object]]
:canonical: checkpoint._backports.filesystem._OverlappingCpuLoader._finish

```{autodoc2-docstring} checkpoint._backports.filesystem._OverlappingCpuLoader._finish
```

````

````{py:method} add(size: int, obj: object) -> None
:canonical: checkpoint._backports.filesystem._OverlappingCpuLoader.add

```{autodoc2-docstring} checkpoint._backports.filesystem._OverlappingCpuLoader.add
```

````

````{py:method} start_loading() -> None
:canonical: checkpoint._backports.filesystem._OverlappingCpuLoader.start_loading

```{autodoc2-docstring} checkpoint._backports.filesystem._OverlappingCpuLoader.start_loading
```

````

````{py:method} values() -> collections.abc.Iterator[tuple[torch.Tensor, object]]
:canonical: checkpoint._backports.filesystem._OverlappingCpuLoader.values

```{autodoc2-docstring} checkpoint._backports.filesystem._OverlappingCpuLoader.values
```

````

`````

`````{py:class} _StorageWriterTransforms(extensions: typing.Optional[collections.abc.Sequence[torch.distributed.checkpoint._extension.StreamTransformExtension]] = None)
:canonical: checkpoint._backports.filesystem._StorageWriterTransforms

```{autodoc2-docstring} checkpoint._backports.filesystem._StorageWriterTransforms
```

```{rubric} Initialization
```

```{autodoc2-docstring} checkpoint._backports.filesystem._StorageWriterTransforms.__init__
```

````{py:method} transform_save_stream(write_item: torch.distributed.checkpoint.planner.WriteItem, raw_stream: io.IOBase) -> tuple[typing.IO[bytes], list[str]]
:canonical: checkpoint._backports.filesystem._StorageWriterTransforms.transform_save_stream

```{autodoc2-docstring} checkpoint._backports.filesystem._StorageWriterTransforms.transform_save_stream
```

````

`````

````{py:function} _item_size(item: torch.distributed.checkpoint.planner.WriteItem) -> int
:canonical: checkpoint._backports.filesystem._item_size

```{autodoc2-docstring} checkpoint._backports.filesystem._item_size
```
````

````{py:function} _split_by_size_and_type(bins: int, items: list[torch.distributed.checkpoint.planner.WriteItem]) -> list[list[torch.distributed.checkpoint.planner.WriteItem]]
:canonical: checkpoint._backports.filesystem._split_by_size_and_type

```{autodoc2-docstring} checkpoint._backports.filesystem._split_by_size_and_type
```
````

````{py:function} _write_item(transforms: checkpoint._backports.filesystem._StorageWriterTransforms, stream: io.IOBase, data: typing.Union[io.BytesIO, torch.Tensor], write_item: torch.distributed.checkpoint.planner.WriteItem, storage_key: str, serialization_format: checkpoint._backports.filesystem.SerializationFormat) -> torch.distributed.checkpoint.storage.WriteResult
:canonical: checkpoint._backports.filesystem._write_item

```{autodoc2-docstring} checkpoint._backports.filesystem._write_item
```
````

````{py:function} _write_files_from_queue(create_stream: typing.Callable, file_queue: queue.Queue, result_queue: queue.Queue, planner: torch.distributed.checkpoint.planner.SavePlanner, transforms: checkpoint._backports.filesystem._StorageWriterTransforms, inflight_threshhold: int, use_fsync: bool, thread_count: int, serialization_format: checkpoint._backports.filesystem.SerializationFormat) -> None
:canonical: checkpoint._backports.filesystem._write_files_from_queue

```{autodoc2-docstring} checkpoint._backports.filesystem._write_files_from_queue
```
````

`````{py:class} FileSystemBase
:canonical: checkpoint._backports.filesystem.FileSystemBase

Bases: {py:obj}`abc.ABC`

````{py:method} create_stream(path: typing.Union[str, os.PathLike], mode: str) -> collections.abc.Generator[io.IOBase, None, None]
:canonical: checkpoint._backports.filesystem.FileSystemBase.create_stream
:abstractmethod:

```{autodoc2-docstring} checkpoint._backports.filesystem.FileSystemBase.create_stream
```

````

````{py:method} concat_path(path: typing.Union[str, os.PathLike], suffix: str) -> typing.Union[str, os.PathLike]
:canonical: checkpoint._backports.filesystem.FileSystemBase.concat_path
:abstractmethod:

```{autodoc2-docstring} checkpoint._backports.filesystem.FileSystemBase.concat_path
```

````

````{py:method} rename(path: typing.Union[str, os.PathLike], new_path: typing.Union[str, os.PathLike]) -> None
:canonical: checkpoint._backports.filesystem.FileSystemBase.rename
:abstractmethod:

```{autodoc2-docstring} checkpoint._backports.filesystem.FileSystemBase.rename
```

````

````{py:method} init_path(path: typing.Union[str, os.PathLike]) -> typing.Union[str, os.PathLike]
:canonical: checkpoint._backports.filesystem.FileSystemBase.init_path
:abstractmethod:

```{autodoc2-docstring} checkpoint._backports.filesystem.FileSystemBase.init_path
```

````

````{py:method} mkdir(path: typing.Union[str, os.PathLike]) -> None
:canonical: checkpoint._backports.filesystem.FileSystemBase.mkdir
:abstractmethod:

```{autodoc2-docstring} checkpoint._backports.filesystem.FileSystemBase.mkdir
```

````

````{py:method} validate_checkpoint_id(checkpoint_id: typing.Union[str, os.PathLike]) -> bool
:canonical: checkpoint._backports.filesystem.FileSystemBase.validate_checkpoint_id
:abstractmethod:
:classmethod:

```{autodoc2-docstring} checkpoint._backports.filesystem.FileSystemBase.validate_checkpoint_id
```

````

````{py:method} exists(path: typing.Union[str, os.PathLike]) -> bool
:canonical: checkpoint._backports.filesystem.FileSystemBase.exists
:abstractmethod:

```{autodoc2-docstring} checkpoint._backports.filesystem.FileSystemBase.exists
```

````

````{py:method} rm_file(path: typing.Union[str, os.PathLike]) -> None
:canonical: checkpoint._backports.filesystem.FileSystemBase.rm_file
:abstractmethod:

```{autodoc2-docstring} checkpoint._backports.filesystem.FileSystemBase.rm_file
```

````

`````

`````{py:class} FileSystem
:canonical: checkpoint._backports.filesystem.FileSystem

Bases: {py:obj}`checkpoint._backports.filesystem.FileSystemBase`

````{py:method} create_stream(path: typing.Union[str, os.PathLike], mode: str) -> collections.abc.Generator[io.IOBase, None, None]
:canonical: checkpoint._backports.filesystem.FileSystem.create_stream

```{autodoc2-docstring} checkpoint._backports.filesystem.FileSystem.create_stream
```

````

````{py:method} concat_path(path: typing.Union[str, os.PathLike], suffix: str) -> typing.Union[str, os.PathLike]
:canonical: checkpoint._backports.filesystem.FileSystem.concat_path

```{autodoc2-docstring} checkpoint._backports.filesystem.FileSystem.concat_path
```

````

````{py:method} init_path(path: typing.Union[str, os.PathLike]) -> typing.Union[str, os.PathLike]
:canonical: checkpoint._backports.filesystem.FileSystem.init_path

```{autodoc2-docstring} checkpoint._backports.filesystem.FileSystem.init_path
```

````

````{py:method} rename(path: typing.Union[str, os.PathLike], new_path: typing.Union[str, os.PathLike]) -> None
:canonical: checkpoint._backports.filesystem.FileSystem.rename

```{autodoc2-docstring} checkpoint._backports.filesystem.FileSystem.rename
```

````

````{py:method} mkdir(path: typing.Union[str, os.PathLike]) -> None
:canonical: checkpoint._backports.filesystem.FileSystem.mkdir

```{autodoc2-docstring} checkpoint._backports.filesystem.FileSystem.mkdir
```

````

````{py:method} validate_checkpoint_id(checkpoint_id: typing.Union[str, os.PathLike]) -> bool
:canonical: checkpoint._backports.filesystem.FileSystem.validate_checkpoint_id
:classmethod:

```{autodoc2-docstring} checkpoint._backports.filesystem.FileSystem.validate_checkpoint_id
```

````

````{py:method} exists(path: typing.Union[str, os.PathLike]) -> bool
:canonical: checkpoint._backports.filesystem.FileSystem.exists

```{autodoc2-docstring} checkpoint._backports.filesystem.FileSystem.exists
```

````

````{py:method} rm_file(path: typing.Union[str, os.PathLike]) -> None
:canonical: checkpoint._backports.filesystem.FileSystem.rm_file

```{autodoc2-docstring} checkpoint._backports.filesystem.FileSystem.rm_file
```

````

````{py:method} ls(path: typing.Union[str, os.PathLike]) -> list[str]
:canonical: checkpoint._backports.filesystem.FileSystem.ls

```{autodoc2-docstring} checkpoint._backports.filesystem.FileSystem.ls
```

````

`````

`````{py:class} _FileSystemWriter(path: typing.Union[str, os.PathLike], single_file_per_rank: bool = True, sync_files: bool = True, thread_count: int = 1, per_thread_copy_ahead: int = 10000000, overwrite: bool = True, _extensions: typing.Optional[collections.abc.Sequence[torch.distributed.checkpoint._extension.StreamTransformExtension]] = None, serialization_format: checkpoint._backports.filesystem.SerializationFormat = SerializationFormat.TORCH_SAVE, *args: typing.Any, **kwargs: typing.Any)
:canonical: checkpoint._backports.filesystem._FileSystemWriter

Bases: {py:obj}`torch.distributed.checkpoint.storage.StorageWriter`

```{autodoc2-docstring} checkpoint._backports.filesystem._FileSystemWriter
```

```{rubric} Initialization
```

```{autodoc2-docstring} checkpoint._backports.filesystem._FileSystemWriter.__init__
```

````{py:method} reset(checkpoint_id: typing.Union[str, os.PathLike, None] = None) -> None
:canonical: checkpoint._backports.filesystem._FileSystemWriter.reset

````

````{py:method} set_up_storage_writer(is_coordinator: bool) -> None
:canonical: checkpoint._backports.filesystem._FileSystemWriter.set_up_storage_writer

````

````{py:method} prepare_local_plan(plan: torch.distributed.checkpoint.planner.SavePlan) -> torch.distributed.checkpoint.planner.SavePlan
:canonical: checkpoint._backports.filesystem._FileSystemWriter.prepare_local_plan

````

````{py:method} prepare_global_plan(plans: list[torch.distributed.checkpoint.planner.SavePlan]) -> list[torch.distributed.checkpoint.planner.SavePlan]
:canonical: checkpoint._backports.filesystem._FileSystemWriter.prepare_global_plan

````

````{py:method} write_data(plan: torch.distributed.checkpoint.planner.SavePlan, planner: torch.distributed.checkpoint.planner.SavePlanner) -> torch.futures.Future[list[torch.distributed.checkpoint.storage.WriteResult]]
:canonical: checkpoint._backports.filesystem._FileSystemWriter.write_data

````

````{py:method} _write_data(planner: torch.distributed.checkpoint.planner.SavePlanner, file_queue: queue.Queue) -> torch.futures.Future[list[torch.distributed.checkpoint.storage.WriteResult]]
:canonical: checkpoint._backports.filesystem._FileSystemWriter._write_data

```{autodoc2-docstring} checkpoint._backports.filesystem._FileSystemWriter._write_data
```

````

````{py:method} finish(metadata: torch.distributed.checkpoint.metadata.Metadata, results: list[list[torch.distributed.checkpoint.storage.WriteResult]]) -> None
:canonical: checkpoint._backports.filesystem._FileSystemWriter.finish

````

````{py:method} storage_meta() -> typing.Optional[torch.distributed.checkpoint.metadata.StorageMeta]
:canonical: checkpoint._backports.filesystem._FileSystemWriter.storage_meta

````

````{py:property} metadata_path
:canonical: checkpoint._backports.filesystem._FileSystemWriter.metadata_path
:type: typing.Union[str, os.PathLike]

```{autodoc2-docstring} checkpoint._backports.filesystem._FileSystemWriter.metadata_path
```

````

````{py:property} checkpoint_id
:canonical: checkpoint._backports.filesystem._FileSystemWriter.checkpoint_id
:type: typing.Union[str, os.PathLike]

```{autodoc2-docstring} checkpoint._backports.filesystem._FileSystemWriter.checkpoint_id
```

````

````{py:method} validate_checkpoint_id(checkpoint_id: typing.Union[str, os.PathLike]) -> bool
:canonical: checkpoint._backports.filesystem._FileSystemWriter.validate_checkpoint_id
:classmethod:

````

`````

`````{py:class} _StorageReaderTransforms(extension_registry: typing.Optional[torch.distributed.checkpoint._extension.ExtensionRegistry] = None)
:canonical: checkpoint._backports.filesystem._StorageReaderTransforms

```{autodoc2-docstring} checkpoint._backports.filesystem._StorageReaderTransforms
```

```{rubric} Initialization
```

```{autodoc2-docstring} checkpoint._backports.filesystem._StorageReaderTransforms.__init__
```

````{py:method} transform_load_stream(read_item: torch.distributed.checkpoint.planner.ReadItem, transform_descriptors: collections.abc.Sequence[str], raw_stream: typing.IO[bytes]) -> typing.IO[bytes]
:canonical: checkpoint._backports.filesystem._StorageReaderTransforms.transform_load_stream

```{autodoc2-docstring} checkpoint._backports.filesystem._StorageReaderTransforms.transform_load_stream
```

````

`````

`````{py:class} FileSystemReader(path: typing.Union[str, os.PathLike], _extension_registry: typing.Optional[torch.distributed.checkpoint._extension.ExtensionRegistry] = None)
:canonical: checkpoint._backports.filesystem.FileSystemReader

Bases: {py:obj}`torch.distributed.checkpoint.storage.StorageReader`

````{py:method} _slice_file(file, sinfo: checkpoint._backports.filesystem._StorageInfo) -> typing.IO[bytes]
:canonical: checkpoint._backports.filesystem.FileSystemReader._slice_file

```{autodoc2-docstring} checkpoint._backports.filesystem.FileSystemReader._slice_file
```

````

````{py:method} reset(checkpoint_id: typing.Union[str, os.PathLike, None] = None) -> None
:canonical: checkpoint._backports.filesystem.FileSystemReader.reset

````

````{py:method} read_data(plan: torch.distributed.checkpoint.planner.LoadPlan, planner: torch.distributed.checkpoint.planner.LoadPlanner) -> torch.futures.Future[None]
:canonical: checkpoint._backports.filesystem.FileSystemReader.read_data

````

````{py:method} read_metadata() -> torch.distributed.checkpoint.metadata.Metadata
:canonical: checkpoint._backports.filesystem.FileSystemReader.read_metadata

````

````{py:method} set_up_storage_reader(metadata: torch.distributed.checkpoint.metadata.Metadata, is_coordinator: bool) -> None
:canonical: checkpoint._backports.filesystem.FileSystemReader.set_up_storage_reader

````

````{py:method} prepare_local_plan(plan: torch.distributed.checkpoint.planner.LoadPlan) -> torch.distributed.checkpoint.planner.LoadPlan
:canonical: checkpoint._backports.filesystem.FileSystemReader.prepare_local_plan

````

````{py:method} prepare_global_plan(plans: list[torch.distributed.checkpoint.planner.LoadPlan]) -> list[torch.distributed.checkpoint.planner.LoadPlan]
:canonical: checkpoint._backports.filesystem.FileSystemReader.prepare_global_plan

````

````{py:property} checkpoint_id
:canonical: checkpoint._backports.filesystem.FileSystemReader.checkpoint_id
:type: typing.Union[str, os.PathLike]

```{autodoc2-docstring} checkpoint._backports.filesystem.FileSystemReader.checkpoint_id
```

````

````{py:method} validate_checkpoint_id(checkpoint_id: typing.Union[str, os.PathLike]) -> bool
:canonical: checkpoint._backports.filesystem.FileSystemReader.validate_checkpoint_id
:classmethod:

````

`````

`````{py:class} FileSystemWriter(path: typing.Union[str, os.PathLike], single_file_per_rank: bool = True, sync_files: bool = True, thread_count: int = 1, per_thread_copy_ahead: int = 10000000, cache_staged_state_dict: bool = False, overwrite: bool = True, _extensions: typing.Optional[collections.abc.Sequence[torch.distributed.checkpoint._extension.StreamTransformExtension]] = None, serialization_format: checkpoint._backports.filesystem.SerializationFormat = SerializationFormat.TORCH_SAVE)
:canonical: checkpoint._backports.filesystem.FileSystemWriter

Bases: {py:obj}`checkpoint._backports.filesystem._FileSystemWriter`, {py:obj}`torch.distributed.checkpoint.staging.BlockingAsyncStager`

```{autodoc2-docstring} checkpoint._backports.filesystem.FileSystemWriter
```

```{rubric} Initialization
```

```{autodoc2-docstring} checkpoint._backports.filesystem.FileSystemWriter.__init__
```

````{py:method} stage(state_dict: torch.distributed.checkpoint.metadata.STATE_DICT_TYPE) -> torch.distributed.checkpoint.metadata.STATE_DICT_TYPE
:canonical: checkpoint._backports.filesystem.FileSystemWriter.stage

```{autodoc2-docstring} checkpoint._backports.filesystem.FileSystemWriter.stage
```

````

`````
