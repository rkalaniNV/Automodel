# {py:mod}`checkpoint._backports._fsspec_filesystem`

```{py:module} checkpoint._backports._fsspec_filesystem
```

```{autodoc2-docstring} checkpoint._backports._fsspec_filesystem
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FileSystem <checkpoint._backports._fsspec_filesystem.FileSystem>`
  -
* - {py:obj}`FsspecWriter <checkpoint._backports._fsspec_filesystem.FsspecWriter>`
  - ```{autodoc2-docstring} checkpoint._backports._fsspec_filesystem.FsspecWriter
    :summary:
    ```
* - {py:obj}`FsspecReader <checkpoint._backports._fsspec_filesystem.FsspecReader>`
  -
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <checkpoint._backports._fsspec_filesystem.__all__>`
  - ```{autodoc2-docstring} checkpoint._backports._fsspec_filesystem.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: checkpoint._backports._fsspec_filesystem.__all__
:value: >
   ['FsspecWriter', 'FsspecReader']

```{autodoc2-docstring} checkpoint._backports._fsspec_filesystem.__all__
```

````

`````{py:class} FileSystem()
:canonical: checkpoint._backports._fsspec_filesystem.FileSystem

Bases: {py:obj}`nemo_automodel.components.checkpoint._backports.filesystem.FileSystemBase`

````{py:method} create_stream(path: typing.Union[str, os.PathLike], mode: str) -> collections.abc.Generator[io.IOBase, None, None]
:canonical: checkpoint._backports._fsspec_filesystem.FileSystem.create_stream

```{autodoc2-docstring} checkpoint._backports._fsspec_filesystem.FileSystem.create_stream
```

````

````{py:method} concat_path(path: typing.Union[str, os.PathLike], suffix: str) -> typing.Union[str, os.PathLike]
:canonical: checkpoint._backports._fsspec_filesystem.FileSystem.concat_path

```{autodoc2-docstring} checkpoint._backports._fsspec_filesystem.FileSystem.concat_path
```

````

````{py:method} init_path(path: typing.Union[str, os.PathLike], **kwargs) -> typing.Union[str, os.PathLike]
:canonical: checkpoint._backports._fsspec_filesystem.FileSystem.init_path

```{autodoc2-docstring} checkpoint._backports._fsspec_filesystem.FileSystem.init_path
```

````

````{py:method} rename(path: typing.Union[str, os.PathLike], new_path: typing.Union[str, os.PathLike]) -> None
:canonical: checkpoint._backports._fsspec_filesystem.FileSystem.rename

```{autodoc2-docstring} checkpoint._backports._fsspec_filesystem.FileSystem.rename
```

````

````{py:method} mkdir(path: typing.Union[str, os.PathLike]) -> None
:canonical: checkpoint._backports._fsspec_filesystem.FileSystem.mkdir

```{autodoc2-docstring} checkpoint._backports._fsspec_filesystem.FileSystem.mkdir
```

````

````{py:method} validate_checkpoint_id(checkpoint_id: typing.Union[str, os.PathLike]) -> bool
:canonical: checkpoint._backports._fsspec_filesystem.FileSystem.validate_checkpoint_id
:classmethod:

```{autodoc2-docstring} checkpoint._backports._fsspec_filesystem.FileSystem.validate_checkpoint_id
```

````

````{py:method} exists(path: typing.Union[str, os.PathLike]) -> bool
:canonical: checkpoint._backports._fsspec_filesystem.FileSystem.exists

```{autodoc2-docstring} checkpoint._backports._fsspec_filesystem.FileSystem.exists
```

````

````{py:method} rm_file(path: typing.Union[str, os.PathLike]) -> None
:canonical: checkpoint._backports._fsspec_filesystem.FileSystem.rm_file

```{autodoc2-docstring} checkpoint._backports._fsspec_filesystem.FileSystem.rm_file
```

````

````{py:method} ls(path: typing.Union[str, os.PathLike]) -> list[str]
:canonical: checkpoint._backports._fsspec_filesystem.FileSystem.ls

```{autodoc2-docstring} checkpoint._backports._fsspec_filesystem.FileSystem.ls
```

````

`````

`````{py:class} FsspecWriter(path: typing.Union[str, os.PathLike], single_file_per_rank: bool = True, sync_files: bool = True, thread_count: int = 1, per_thread_copy_ahead: int = 10000000, overwrite: bool = True, _extensions: typing.Optional[collections.abc.Sequence[torch.distributed.checkpoint._extension.StreamTransformExtension]] = None, serialization_format: nemo_automodel.components.checkpoint._backports.filesystem.SerializationFormat = SerializationFormat.TORCH_SAVE, **kwargs)
:canonical: checkpoint._backports._fsspec_filesystem.FsspecWriter

Bases: {py:obj}`nemo_automodel.components.checkpoint._backports.filesystem.FileSystemWriter`

```{autodoc2-docstring} checkpoint._backports._fsspec_filesystem.FsspecWriter
```

```{rubric} Initialization
```

```{autodoc2-docstring} checkpoint._backports._fsspec_filesystem.FsspecWriter.__init__
```

````{py:method} validate_checkpoint_id(checkpoint_id: typing.Union[str, os.PathLike]) -> bool
:canonical: checkpoint._backports._fsspec_filesystem.FsspecWriter.validate_checkpoint_id
:classmethod:

````

`````

`````{py:class} FsspecReader(path: typing.Union[str, os.PathLike], **kwargs)
:canonical: checkpoint._backports._fsspec_filesystem.FsspecReader

Bases: {py:obj}`nemo_automodel.components.checkpoint._backports.filesystem.FileSystemReader`

````{py:method} validate_checkpoint_id(checkpoint_id: typing.Union[str, os.PathLike]) -> bool
:canonical: checkpoint._backports._fsspec_filesystem.FsspecReader.validate_checkpoint_id
:classmethod:

````

`````
