# {py:mod}`nemo_automodel.components.distributed.parallelizer`

```{py:module} nemo_automodel.components.distributed.parallelizer
```

```{autodoc2-docstring} nemo_automodel.components.distributed.parallelizer
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`apply_fsdp2_sharding_recursively <nemo_automodel.components.distributed.parallelizer.apply_fsdp2_sharding_recursively>`
  - ```{autodoc2-docstring} nemo_automodel.components.distributed.parallelizer.apply_fsdp2_sharding_recursively
    :summary:
    ```
* - {py:obj}`get_hf_tp_shard_plan <nemo_automodel.components.distributed.parallelizer.get_hf_tp_shard_plan>`
  - ```{autodoc2-docstring} nemo_automodel.components.distributed.parallelizer.get_hf_tp_shard_plan
    :summary:
    ```
* - {py:obj}`import_class_from_path <nemo_automodel.components.distributed.parallelizer.import_class_from_path>`
  - ```{autodoc2-docstring} nemo_automodel.components.distributed.parallelizer.import_class_from_path
    :summary:
    ```
* - {py:obj}`import_classes_from_paths <nemo_automodel.components.distributed.parallelizer.import_classes_from_paths>`
  - ```{autodoc2-docstring} nemo_automodel.components.distributed.parallelizer.import_classes_from_paths
    :summary:
    ```
* - {py:obj}`translate_to_torch_parallel_style <nemo_automodel.components.distributed.parallelizer.translate_to_torch_parallel_style>`
  - ```{autodoc2-docstring} nemo_automodel.components.distributed.parallelizer.translate_to_torch_parallel_style
    :summary:
    ```
* - {py:obj}`validate_tp_mesh <nemo_automodel.components.distributed.parallelizer.validate_tp_mesh>`
  - ```{autodoc2-docstring} nemo_automodel.components.distributed.parallelizer.validate_tp_mesh
    :summary:
    ```
* - {py:obj}`get_lm_ac_layers <nemo_automodel.components.distributed.parallelizer.get_lm_ac_layers>`
  - ```{autodoc2-docstring} nemo_automodel.components.distributed.parallelizer.get_lm_ac_layers
    :summary:
    ```
* - {py:obj}`_get_parallel_plan <nemo_automodel.components.distributed.parallelizer._get_parallel_plan>`
  - ```{autodoc2-docstring} nemo_automodel.components.distributed.parallelizer._get_parallel_plan
    :summary:
    ```
* - {py:obj}`fsdp2_strategy_parallelize <nemo_automodel.components.distributed.parallelizer.fsdp2_strategy_parallelize>`
  - ```{autodoc2-docstring} nemo_automodel.components.distributed.parallelizer.fsdp2_strategy_parallelize
    :summary:
    ```
* - {py:obj}`nvfsdp_strategy_parallelize <nemo_automodel.components.distributed.parallelizer.nvfsdp_strategy_parallelize>`
  - ```{autodoc2-docstring} nemo_automodel.components.distributed.parallelizer.nvfsdp_strategy_parallelize
    :summary:
    ```
* - {py:obj}`unshard_fsdp2_model <nemo_automodel.components.distributed.parallelizer.unshard_fsdp2_model>`
  - ```{autodoc2-docstring} nemo_automodel.components.distributed.parallelizer.unshard_fsdp2_model
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HAVE_NVFSDP <nemo_automodel.components.distributed.parallelizer.HAVE_NVFSDP>`
  - ```{autodoc2-docstring} nemo_automodel.components.distributed.parallelizer.HAVE_NVFSDP
    :summary:
    ```
````

### API

````{py:data} HAVE_NVFSDP
:canonical: nemo_automodel.components.distributed.parallelizer.HAVE_NVFSDP
:value: >
   False

```{autodoc2-docstring} nemo_automodel.components.distributed.parallelizer.HAVE_NVFSDP
```

````

````{py:function} apply_fsdp2_sharding_recursively(module: torch.nn.Module, mesh: torch.distributed.device_mesh.DeviceMesh, mp_policy: typing.Optional[torch.distributed.fsdp.MixedPrecisionPolicy], offload_policy: typing.Optional[torch.distributed.fsdp.OffloadPolicy] = None) -> None
:canonical: nemo_automodel.components.distributed.parallelizer.apply_fsdp2_sharding_recursively

```{autodoc2-docstring} nemo_automodel.components.distributed.parallelizer.apply_fsdp2_sharding_recursively
```
````

````{py:function} get_hf_tp_shard_plan(model)
:canonical: nemo_automodel.components.distributed.parallelizer.get_hf_tp_shard_plan

```{autodoc2-docstring} nemo_automodel.components.distributed.parallelizer.get_hf_tp_shard_plan
```
````

````{py:function} import_class_from_path(name: str) -> typing.Any
:canonical: nemo_automodel.components.distributed.parallelizer.import_class_from_path

```{autodoc2-docstring} nemo_automodel.components.distributed.parallelizer.import_class_from_path
```
````

````{py:function} import_classes_from_paths(class_paths: typing.List[str])
:canonical: nemo_automodel.components.distributed.parallelizer.import_classes_from_paths

```{autodoc2-docstring} nemo_automodel.components.distributed.parallelizer.import_classes_from_paths
```
````

````{py:function} translate_to_torch_parallel_style(style: str)
:canonical: nemo_automodel.components.distributed.parallelizer.translate_to_torch_parallel_style

```{autodoc2-docstring} nemo_automodel.components.distributed.parallelizer.translate_to_torch_parallel_style
```
````

````{py:function} validate_tp_mesh(model, tp_mesh)
:canonical: nemo_automodel.components.distributed.parallelizer.validate_tp_mesh

```{autodoc2-docstring} nemo_automodel.components.distributed.parallelizer.validate_tp_mesh
```
````

````{py:function} get_lm_ac_layers(model: torch.nn.Module) -> typing.List[torch.nn.Module]
:canonical: nemo_automodel.components.distributed.parallelizer.get_lm_ac_layers

```{autodoc2-docstring} nemo_automodel.components.distributed.parallelizer.get_lm_ac_layers
```
````

````{py:function} _get_parallel_plan(model: torch.nn.Module, sequence_parallel: bool = False, tp_shard_plan: typing.Optional[typing.Union[typing.Dict[str, torch.distributed.tensor.parallel.ParallelStyle], str]] = None) -> typing.Dict[str, torch.distributed.tensor.parallel.ParallelStyle]
:canonical: nemo_automodel.components.distributed.parallelizer._get_parallel_plan

```{autodoc2-docstring} nemo_automodel.components.distributed.parallelizer._get_parallel_plan
```
````

````{py:function} fsdp2_strategy_parallelize(model, device_mesh: torch.distributed.device_mesh.DeviceMesh, mp_policy: typing.Optional[torch.distributed.fsdp.MixedPrecisionPolicy] = None, offload_policy: typing.Optional[torch.distributed.fsdp.OffloadPolicy] = None, sequence_parallel: bool = False, activation_checkpointing: bool = False, tp_shard_plan: typing.Optional[typing.Union[typing.Dict[str, torch.distributed.tensor.parallel.ParallelStyle], str]] = None, dp_replicate_mesh_name: str = 'dp_replicate', dp_shard_cp_mesh_name: str = 'dp_shard_cp', tp_mesh_name: str = 'tp')
:canonical: nemo_automodel.components.distributed.parallelizer.fsdp2_strategy_parallelize

```{autodoc2-docstring} nemo_automodel.components.distributed.parallelizer.fsdp2_strategy_parallelize
```
````

````{py:function} nvfsdp_strategy_parallelize(model, device_mesh: torch.distributed.device_mesh.DeviceMesh, optimizer=None, nvfsdp_unit_modules: typing.Optional[typing.List[str]] = None, tp_shard_plan: typing.Optional[typing.Dict[str, typing.Union[torch.distributed.tensor.parallel.RowwiseParallel, torch.distributed.tensor.parallel.ColwiseParallel, torch.distributed.tensor.parallel.SequenceParallel]]] = None, data_parallel_sharding_strategy: str = 'optim_grads_params', init_nvfsdp_with_meta_device: bool = False, grad_reduce_in_fp32: bool = False, preserve_fp32_weights: bool = False, overlap_grad_reduce: bool = True, overlap_param_gather: bool = True, check_for_nan_in_grad: bool = True, average_in_collective: bool = False, disable_bucketing: bool = False, calculate_per_token_loss: bool = False, keep_fp8_transpose_cache_when_using_custom_fsdp: bool = False, nccl_ub: bool = False, fsdp_double_buffer: bool = False, dp_mesh_name: str = 'dp', cp_mesh_name: str = 'cp', tp_mesh_name: str = 'tp')
:canonical: nemo_automodel.components.distributed.parallelizer.nvfsdp_strategy_parallelize

```{autodoc2-docstring} nemo_automodel.components.distributed.parallelizer.nvfsdp_strategy_parallelize
```
````

````{py:function} unshard_fsdp2_model(model: torch.nn.Module) -> typing.Generator[None, None, None]
:canonical: nemo_automodel.components.distributed.parallelizer.unshard_fsdp2_model

```{autodoc2-docstring} nemo_automodel.components.distributed.parallelizer.unshard_fsdp2_model
```
````
