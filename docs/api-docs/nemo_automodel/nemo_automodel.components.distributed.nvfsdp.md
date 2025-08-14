# {py:mod}`nemo_automodel.components.distributed.nvfsdp`

```{py:module} nemo_automodel.components.distributed.nvfsdp
```

```{autodoc2-docstring} nemo_automodel.components.distributed.nvfsdp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NVFSDPManager <nemo_automodel.components.distributed.nvfsdp.NVFSDPManager>`
  - ```{autodoc2-docstring} nemo_automodel.components.distributed.nvfsdp.NVFSDPManager
    :summary:
    ```
````

### API

`````{py:class} NVFSDPManager
:canonical: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager

```{autodoc2-docstring} nemo_automodel.components.distributed.nvfsdp.NVFSDPManager
```

````{py:attribute} dp_size
:canonical: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.dp_size
:type: typing.Optional[int]
:value: >
   'field(...)'

```{autodoc2-docstring} nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.dp_size
```

````

````{py:attribute} tp_size
:canonical: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.tp_size
:type: typing.Optional[int]
:value: >
   'field(...)'

```{autodoc2-docstring} nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.tp_size
```

````

````{py:attribute} cp_size
:canonical: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.cp_size
:type: typing.Optional[int]
:value: >
   'field(...)'

```{autodoc2-docstring} nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.cp_size
```

````

````{py:attribute} sequence_parallel
:canonical: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.sequence_parallel
:type: typing.Optional[bool]
:value: >
   'field(...)'

```{autodoc2-docstring} nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.sequence_parallel
```

````

````{py:attribute} backend
:canonical: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.backend
:type: typing.Optional[str]
:value: >
   'field(...)'

```{autodoc2-docstring} nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.backend
```

````

````{py:attribute} world_size
:canonical: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.world_size
:type: typing.Optional[int]
:value: >
   'field(...)'

```{autodoc2-docstring} nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.world_size
```

````

````{py:attribute} nvfsdp_unit_modules
:canonical: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.nvfsdp_unit_modules
:type: typing.Optional[typing.List[str]]
:value: >
   'field(...)'

```{autodoc2-docstring} nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.nvfsdp_unit_modules
```

````

````{py:attribute} data_parallel_sharding_strategy
:canonical: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.data_parallel_sharding_strategy
:type: typing.Optional[str]
:value: >
   'field(...)'

```{autodoc2-docstring} nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.data_parallel_sharding_strategy
```

````

````{py:attribute} init_nvfsdp_with_meta_device
:canonical: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.init_nvfsdp_with_meta_device
:type: typing.Optional[bool]
:value: >
   'field(...)'

```{autodoc2-docstring} nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.init_nvfsdp_with_meta_device
```

````

````{py:attribute} grad_reduce_in_fp32
:canonical: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.grad_reduce_in_fp32
:type: typing.Optional[bool]
:value: >
   'field(...)'

```{autodoc2-docstring} nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.grad_reduce_in_fp32
```

````

````{py:attribute} preserve_fp32_weights
:canonical: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.preserve_fp32_weights
:type: typing.Optional[bool]
:value: >
   'field(...)'

```{autodoc2-docstring} nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.preserve_fp32_weights
```

````

````{py:attribute} overlap_grad_reduce
:canonical: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.overlap_grad_reduce
:type: typing.Optional[bool]
:value: >
   'field(...)'

```{autodoc2-docstring} nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.overlap_grad_reduce
```

````

````{py:attribute} overlap_param_gather
:canonical: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.overlap_param_gather
:type: typing.Optional[bool]
:value: >
   'field(...)'

```{autodoc2-docstring} nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.overlap_param_gather
```

````

````{py:attribute} check_for_nan_in_grad
:canonical: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.check_for_nan_in_grad
:type: typing.Optional[bool]
:value: >
   'field(...)'

```{autodoc2-docstring} nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.check_for_nan_in_grad
```

````

````{py:attribute} average_in_collective
:canonical: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.average_in_collective
:type: typing.Optional[bool]
:value: >
   'field(...)'

```{autodoc2-docstring} nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.average_in_collective
```

````

````{py:attribute} disable_bucketing
:canonical: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.disable_bucketing
:type: typing.Optional[bool]
:value: >
   'field(...)'

```{autodoc2-docstring} nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.disable_bucketing
```

````

````{py:attribute} calculate_per_token_loss
:canonical: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.calculate_per_token_loss
:type: typing.Optional[bool]
:value: >
   'field(...)'

```{autodoc2-docstring} nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.calculate_per_token_loss
```

````

````{py:attribute} keep_fp8_transpose_cache_when_using_custom_fsdp
:canonical: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.keep_fp8_transpose_cache_when_using_custom_fsdp
:type: typing.Optional[bool]
:value: >
   'field(...)'

```{autodoc2-docstring} nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.keep_fp8_transpose_cache_when_using_custom_fsdp
```

````

````{py:attribute} nccl_ub
:canonical: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.nccl_ub
:type: typing.Optional[bool]
:value: >
   'field(...)'

```{autodoc2-docstring} nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.nccl_ub
```

````

````{py:attribute} fsdp_double_buffer
:canonical: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.fsdp_double_buffer
:type: typing.Optional[bool]
:value: >
   'field(...)'

```{autodoc2-docstring} nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.fsdp_double_buffer
```

````

````{py:method} __post_init__()
:canonical: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.__post_init__

```{autodoc2-docstring} nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.__post_init__
```

````

````{py:method} _setup_distributed()
:canonical: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager._setup_distributed

```{autodoc2-docstring} nemo_automodel.components.distributed.nvfsdp.NVFSDPManager._setup_distributed
```

````

````{py:method} parallelize(model, optimizer=None, use_hf_tp_plan=False)
:canonical: nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.parallelize

```{autodoc2-docstring} nemo_automodel.components.distributed.nvfsdp.NVFSDPManager.parallelize
```

````

`````
