# {py:mod}`nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors`

```{py:module} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors
```

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_FqnData <nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._FqnData>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._FqnData
    :summary:
    ```
* - {py:obj}`_OutputFileData <nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._OutputFileData>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._OutputFileData
    :summary:
    ```
* - {py:obj}`_InputFileData <nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._InputFileData>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._InputFileData
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_parse_input_metadata <nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._parse_input_metadata>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._parse_input_metadata
    :summary:
    ```
* - {py:obj}`_write_metadata <nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_metadata>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_metadata
    :summary:
    ```
* - {py:obj}`_read_tensor_data_mmap <nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._read_tensor_data_mmap>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._read_tensor_data_mmap
    :summary:
    ```
* - {py:obj}`_process_output_file <nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._process_output_file>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._process_output_file
    :summary:
    ```
* - {py:obj}`_write_data <nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_data>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_data
    :summary:
    ```
* - {py:obj}`_write_row_wise_tensor <nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_row_wise_tensor>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_row_wise_tensor
    :summary:
    ```
* - {py:obj}`_write_column_wise_tensor <nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_column_wise_tensor>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_column_wise_tensor
    :summary:
    ```
* - {py:obj}`_write_element_by_element <nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_element_by_element>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_element_by_element
    :summary:
    ```
* - {py:obj}`_write_sub_tensor_to_file_optimized <nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_sub_tensor_to_file_optimized>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_sub_tensor_to_file_optimized
    :summary:
    ```
* - {py:obj}`_write_row_wise_tensor_optimized <nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_row_wise_tensor_optimized>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_row_wise_tensor_optimized
    :summary:
    ```
* - {py:obj}`_write_sub_tensor_to_file <nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_sub_tensor_to_file>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_sub_tensor_to_file
    :summary:
    ```
* - {py:obj}`_write_overall_metadata_file <nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_overall_metadata_file>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_overall_metadata_file
    :summary:
    ```
* - {py:obj}`consolidate_safetensors_files <nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors.consolidate_safetensors_files>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors.consolidate_safetensors_files
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors.logger>`
  - ```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors.logger
:type: logging.Logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors.logger
```

````

`````{py:class} _FqnData
:canonical: nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._FqnData

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._FqnData
```

````{py:attribute} offset_in_file
:canonical: nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._FqnData.offset_in_file
:type: int
:value: >
   0

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._FqnData.offset_in_file
```

````

````{py:attribute} shape_in_file
:canonical: nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._FqnData.shape_in_file
:type: list[int]
:value: >
   'field(...)'

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._FqnData.shape_in_file
```

````

````{py:attribute} dtype_size
:canonical: nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._FqnData.dtype_size
:type: int
:value: >
   0

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._FqnData.dtype_size
```

````

````{py:attribute} dtype_str
:canonical: nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._FqnData.dtype_str
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._FqnData.dtype_str
```

````

`````

`````{py:class} _OutputFileData
:canonical: nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._OutputFileData

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._OutputFileData
```

````{py:attribute} metadata_size
:canonical: nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._OutputFileData.metadata_size
:type: int
:value: >
   0

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._OutputFileData.metadata_size
```

````

````{py:attribute} fqn_data
:canonical: nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._OutputFileData.fqn_data
:type: dict[str, nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._FqnData]
:value: >
   'field(...)'

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._OutputFileData.fqn_data
```

````

`````

`````{py:class} _InputFileData
:canonical: nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._InputFileData

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._InputFileData
```

````{py:attribute} metadata_size
:canonical: nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._InputFileData.metadata_size
:type: int
:value: >
   0

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._InputFileData.metadata_size
```

````

````{py:attribute} metadata
:canonical: nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._InputFileData.metadata
:type: typing.Any
:value: >
   None

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._InputFileData.metadata
```

````

`````

````{py:function} _parse_input_metadata(input_files_data: dict[str, nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._InputFileData], output_files_data: dict[str, nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._OutputFileData]) -> None
:canonical: nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._parse_input_metadata

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._parse_input_metadata
```
````

````{py:function} _write_metadata(fs: fsspec.AbstractFileSystem, output_files_data: dict[str, nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._OutputFileData]) -> None
:canonical: nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_metadata

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_metadata
```
````

````{py:function} _read_tensor_data_mmap(input_fs: fsspec.AbstractFileSystem, file_path: str, start_offset: int, end_offset: int, metadata_size: int) -> bytes
:canonical: nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._read_tensor_data_mmap

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._read_tensor_data_mmap
```
````

````{py:function} _process_output_file(input_fs: fsspec.AbstractFileSystem, output_fs: fsspec.AbstractFileSystem, output_file: str, output_data: nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._OutputFileData, input_files_data: dict[str, nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._InputFileData]) -> None
:canonical: nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._process_output_file

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._process_output_file
```
````

````{py:function} _write_data(input_fs: fsspec.AbstractFileSystem, output_fs: fsspec.AbstractFileSystem, input_files_data: dict[str, nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._InputFileData], output_files_data: dict[str, nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._OutputFileData], num_threads: int = 1) -> None
:canonical: nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_data

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_data
```
````

````{py:function} _write_row_wise_tensor(fs: fsspec.AbstractFileSystem, sub_tensor_bytes: bytearray, element_size: int, full_tensor_strides: list[int], sub_tensor_strides: list[int], sub_tensor_offsets: list[int], sub_tensor_shape: list[int], output_file_path: str, output_start_byte: int) -> None
:canonical: nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_row_wise_tensor

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_row_wise_tensor
```
````

````{py:function} _write_column_wise_tensor(fs: fsspec.AbstractFileSystem, sub_tensor_bytes: bytearray, element_size: int, tensor_shape: list[int], sub_tensor_offsets: list[int], sub_tensor_shape: list[int], output_file_path: str, output_start_byte: int) -> None
:canonical: nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_column_wise_tensor

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_column_wise_tensor
```
````

````{py:function} _write_element_by_element(fs: fsspec.AbstractFileSystem, sub_tensor_bytes: bytearray, element_size: int, tensor_shape: list[int], full_tensor_strides: list[int], sub_tensor_strides: list[int], sub_tensor_offsets: list[int], sub_tensor_shape: list[int], output_file_path: str, output_start_byte: int) -> None
:canonical: nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_element_by_element

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_element_by_element
```
````

````{py:function} _write_sub_tensor_to_file_optimized(fs: fsspec.AbstractFileSystem, sub_tensor_bytes: bytes, element_size: int, tensor_shape: list[int], sub_tensor_offsets: list[int], sub_tensor_shape: list[int], output_file_path: str, output_start_byte: int) -> None
:canonical: nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_sub_tensor_to_file_optimized

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_sub_tensor_to_file_optimized
```
````

````{py:function} _write_row_wise_tensor_optimized(fs: fsspec.AbstractFileSystem, sub_tensor_bytes: bytes, element_size: int, tensor_shape: list[int], sub_tensor_offsets: list[int], sub_tensor_shape: list[int], output_file_path: str, output_start_byte: int) -> None
:canonical: nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_row_wise_tensor_optimized

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_row_wise_tensor_optimized
```
````

````{py:function} _write_sub_tensor_to_file(fs: fsspec.AbstractFileSystem, sub_tensor_bytes: bytearray, element_size: int, tensor_shape: list[int], sub_tensor_offsets: list[int], sub_tensor_shape: list[int], output_file_path: str, output_start_byte: int) -> None
:canonical: nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_sub_tensor_to_file

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_sub_tensor_to_file
```
````

````{py:function} _write_overall_metadata_file(fs: fsspec.AbstractFileSystem, output_dir: str, output_files_data: dict[str, nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._OutputFileData]) -> None
:canonical: nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_overall_metadata_file

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._write_overall_metadata_file
```
````

````{py:function} consolidate_safetensors_files(input_dir: str, output_dir: str, fqn_to_index_mapping: typing.Optional[dict[str, int]] = None, num_threads: int = 1) -> None
:canonical: nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors.consolidate_safetensors_files

```{autodoc2-docstring} nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors.consolidate_safetensors_files
```
````
