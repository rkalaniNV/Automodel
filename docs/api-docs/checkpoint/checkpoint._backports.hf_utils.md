# {py:mod}`checkpoint._backports.hf_utils`

```{py:module} checkpoint._backports.hf_utils
```

```{autodoc2-docstring} checkpoint._backports.hf_utils
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_HFStorageInfo <checkpoint._backports.hf_utils._HFStorageInfo>`
  - ```{autodoc2-docstring} checkpoint._backports.hf_utils._HFStorageInfo
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_gen_file_name <checkpoint._backports.hf_utils._gen_file_name>`
  - ```{autodoc2-docstring} checkpoint._backports.hf_utils._gen_file_name
    :summary:
    ```
* - {py:obj}`_get_safetensors_file_metadata <checkpoint._backports.hf_utils._get_safetensors_file_metadata>`
  - ```{autodoc2-docstring} checkpoint._backports.hf_utils._get_safetensors_file_metadata
    :summary:
    ```
* - {py:obj}`_get_dtype <checkpoint._backports.hf_utils._get_dtype>`
  - ```{autodoc2-docstring} checkpoint._backports.hf_utils._get_dtype
    :summary:
    ```
* - {py:obj}`_get_dcp_custom_metadata <checkpoint._backports.hf_utils._get_dcp_custom_metadata>`
  - ```{autodoc2-docstring} checkpoint._backports.hf_utils._get_dcp_custom_metadata
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_metadata_fn <checkpoint._backports.hf_utils._metadata_fn>`
  - ```{autodoc2-docstring} checkpoint._backports.hf_utils._metadata_fn
    :summary:
    ```
* - {py:obj}`FILE_NAME <checkpoint._backports.hf_utils.FILE_NAME>`
  - ```{autodoc2-docstring} checkpoint._backports.hf_utils.FILE_NAME
    :summary:
    ```
* - {py:obj}`SHARDED_FILE_NAME <checkpoint._backports.hf_utils.SHARDED_FILE_NAME>`
  - ```{autodoc2-docstring} checkpoint._backports.hf_utils.SHARDED_FILE_NAME
    :summary:
    ```
* - {py:obj}`SUFFIX <checkpoint._backports.hf_utils.SUFFIX>`
  - ```{autodoc2-docstring} checkpoint._backports.hf_utils.SUFFIX
    :summary:
    ```
* - {py:obj}`CUSTOM_METADATA_KEY <checkpoint._backports.hf_utils.CUSTOM_METADATA_KEY>`
  - ```{autodoc2-docstring} checkpoint._backports.hf_utils.CUSTOM_METADATA_KEY
    :summary:
    ```
* - {py:obj}`DEFAULT_EXTRA_METADATA_KEY <checkpoint._backports.hf_utils.DEFAULT_EXTRA_METADATA_KEY>`
  - ```{autodoc2-docstring} checkpoint._backports.hf_utils.DEFAULT_EXTRA_METADATA_KEY
    :summary:
    ```
* - {py:obj}`SAVED_OFFSETS_KEY <checkpoint._backports.hf_utils.SAVED_OFFSETS_KEY>`
  - ```{autodoc2-docstring} checkpoint._backports.hf_utils.SAVED_OFFSETS_KEY
    :summary:
    ```
* - {py:obj}`SHAPE_KEY <checkpoint._backports.hf_utils.SHAPE_KEY>`
  - ```{autodoc2-docstring} checkpoint._backports.hf_utils.SHAPE_KEY
    :summary:
    ```
* - {py:obj}`DATA_KEY <checkpoint._backports.hf_utils.DATA_KEY>`
  - ```{autodoc2-docstring} checkpoint._backports.hf_utils.DATA_KEY
    :summary:
    ```
* - {py:obj}`DTYPE_KEY <checkpoint._backports.hf_utils.DTYPE_KEY>`
  - ```{autodoc2-docstring} checkpoint._backports.hf_utils.DTYPE_KEY
    :summary:
    ```
* - {py:obj}`DATA_OFFSETS_KEY <checkpoint._backports.hf_utils.DATA_OFFSETS_KEY>`
  - ```{autodoc2-docstring} checkpoint._backports.hf_utils.DATA_OFFSETS_KEY
    :summary:
    ```
* - {py:obj}`DTYPE_MAP <checkpoint._backports.hf_utils.DTYPE_MAP>`
  - ```{autodoc2-docstring} checkpoint._backports.hf_utils.DTYPE_MAP
    :summary:
    ```
* - {py:obj}`HF_DCP_VERSION <checkpoint._backports.hf_utils.HF_DCP_VERSION>`
  - ```{autodoc2-docstring} checkpoint._backports.hf_utils.HF_DCP_VERSION
    :summary:
    ```
* - {py:obj}`DCP_VERSION_KEY <checkpoint._backports.hf_utils.DCP_VERSION_KEY>`
  - ```{autodoc2-docstring} checkpoint._backports.hf_utils.DCP_VERSION_KEY
    :summary:
    ```
* - {py:obj}`DCP_SHARDING_INFO_KEY <checkpoint._backports.hf_utils.DCP_SHARDING_INFO_KEY>`
  - ```{autodoc2-docstring} checkpoint._backports.hf_utils.DCP_SHARDING_INFO_KEY
    :summary:
    ```
````

### API

````{py:data} _metadata_fn
:canonical: checkpoint._backports.hf_utils._metadata_fn
:type: str
:value: >
   'model.safetensors.index.json'

```{autodoc2-docstring} checkpoint._backports.hf_utils._metadata_fn
```

````

````{py:data} FILE_NAME
:canonical: checkpoint._backports.hf_utils.FILE_NAME
:value: >
   'model-{cpt_idx}-of-{num_files}'

```{autodoc2-docstring} checkpoint._backports.hf_utils.FILE_NAME
```

````

````{py:data} SHARDED_FILE_NAME
:canonical: checkpoint._backports.hf_utils.SHARDED_FILE_NAME
:value: >
   'shard-{shard_idx}-model-{cpt_idx}-of-{num_files}'

```{autodoc2-docstring} checkpoint._backports.hf_utils.SHARDED_FILE_NAME
```

````

````{py:data} SUFFIX
:canonical: checkpoint._backports.hf_utils.SUFFIX
:value: >
   '.safetensors'

```{autodoc2-docstring} checkpoint._backports.hf_utils.SUFFIX
```

````

````{py:data} CUSTOM_METADATA_KEY
:canonical: checkpoint._backports.hf_utils.CUSTOM_METADATA_KEY
:value: >
   'DCP_SHARDING_INFO'

```{autodoc2-docstring} checkpoint._backports.hf_utils.CUSTOM_METADATA_KEY
```

````

````{py:data} DEFAULT_EXTRA_METADATA_KEY
:canonical: checkpoint._backports.hf_utils.DEFAULT_EXTRA_METADATA_KEY
:value: >
   '__metadata__'

```{autodoc2-docstring} checkpoint._backports.hf_utils.DEFAULT_EXTRA_METADATA_KEY
```

````

````{py:data} SAVED_OFFSETS_KEY
:canonical: checkpoint._backports.hf_utils.SAVED_OFFSETS_KEY
:value: >
   'saved_offsets'

```{autodoc2-docstring} checkpoint._backports.hf_utils.SAVED_OFFSETS_KEY
```

````

````{py:data} SHAPE_KEY
:canonical: checkpoint._backports.hf_utils.SHAPE_KEY
:value: >
   'shape'

```{autodoc2-docstring} checkpoint._backports.hf_utils.SHAPE_KEY
```

````

````{py:data} DATA_KEY
:canonical: checkpoint._backports.hf_utils.DATA_KEY
:value: >
   'data'

```{autodoc2-docstring} checkpoint._backports.hf_utils.DATA_KEY
```

````

````{py:data} DTYPE_KEY
:canonical: checkpoint._backports.hf_utils.DTYPE_KEY
:value: >
   'dtype'

```{autodoc2-docstring} checkpoint._backports.hf_utils.DTYPE_KEY
```

````

````{py:data} DATA_OFFSETS_KEY
:canonical: checkpoint._backports.hf_utils.DATA_OFFSETS_KEY
:value: >
   'data_offsets'

```{autodoc2-docstring} checkpoint._backports.hf_utils.DATA_OFFSETS_KEY
```

````

````{py:data} DTYPE_MAP
:canonical: checkpoint._backports.hf_utils.DTYPE_MAP
:value: >
   None

```{autodoc2-docstring} checkpoint._backports.hf_utils.DTYPE_MAP
```

````

````{py:data} HF_DCP_VERSION
:canonical: checkpoint._backports.hf_utils.HF_DCP_VERSION
:type: float
:value: >
   1.0

```{autodoc2-docstring} checkpoint._backports.hf_utils.HF_DCP_VERSION
```

````

````{py:data} DCP_VERSION_KEY
:canonical: checkpoint._backports.hf_utils.DCP_VERSION_KEY
:value: >
   'DCP_VERSION'

```{autodoc2-docstring} checkpoint._backports.hf_utils.DCP_VERSION_KEY
```

````

````{py:data} DCP_SHARDING_INFO_KEY
:canonical: checkpoint._backports.hf_utils.DCP_SHARDING_INFO_KEY
:value: >
   'DCP_SHARDING_INFO'

```{autodoc2-docstring} checkpoint._backports.hf_utils.DCP_SHARDING_INFO_KEY
```

````

`````{py:class} _HFStorageInfo
:canonical: checkpoint._backports.hf_utils._HFStorageInfo

```{autodoc2-docstring} checkpoint._backports.hf_utils._HFStorageInfo
```

````{py:attribute} relative_path
:canonical: checkpoint._backports.hf_utils._HFStorageInfo.relative_path
:type: str
:value: >
   None

```{autodoc2-docstring} checkpoint._backports.hf_utils._HFStorageInfo.relative_path
```

````

````{py:attribute} offset
:canonical: checkpoint._backports.hf_utils._HFStorageInfo.offset
:type: int
:value: >
   None

```{autodoc2-docstring} checkpoint._backports.hf_utils._HFStorageInfo.offset
```

````

````{py:attribute} length
:canonical: checkpoint._backports.hf_utils._HFStorageInfo.length
:type: int
:value: >
   None

```{autodoc2-docstring} checkpoint._backports.hf_utils._HFStorageInfo.length
```

````

````{py:attribute} shape
:canonical: checkpoint._backports.hf_utils._HFStorageInfo.shape
:type: torch.Size
:value: >
   None

```{autodoc2-docstring} checkpoint._backports.hf_utils._HFStorageInfo.shape
```

````

````{py:attribute} dtype
:canonical: checkpoint._backports.hf_utils._HFStorageInfo.dtype
:type: torch.dtype
:value: >
   None

```{autodoc2-docstring} checkpoint._backports.hf_utils._HFStorageInfo.dtype
```

````

````{py:method} __getstate__()
:canonical: checkpoint._backports.hf_utils._HFStorageInfo.__getstate__

````

`````

````{py:function} _gen_file_name(index: int, largest_index: int, shard_index: typing.Optional[int] = None) -> str
:canonical: checkpoint._backports.hf_utils._gen_file_name

```{autodoc2-docstring} checkpoint._backports.hf_utils._gen_file_name
```
````

````{py:function} _get_safetensors_file_metadata(file_bytes: io.IOBase) -> tuple[typing.Any, int]
:canonical: checkpoint._backports.hf_utils._get_safetensors_file_metadata

```{autodoc2-docstring} checkpoint._backports.hf_utils._get_safetensors_file_metadata
```
````

````{py:function} _get_dtype(dtype_str: str) -> torch.dtype
:canonical: checkpoint._backports.hf_utils._get_dtype

```{autodoc2-docstring} checkpoint._backports.hf_utils._get_dtype
```
````

````{py:function} _get_dcp_custom_metadata(metadata: typing.Any) -> typing.Optional[typing.Any]
:canonical: checkpoint._backports.hf_utils._get_dcp_custom_metadata

```{autodoc2-docstring} checkpoint._backports.hf_utils._get_dcp_custom_metadata
```
````
