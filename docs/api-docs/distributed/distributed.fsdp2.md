# {py:mod}`distributed.fsdp2`

```{py:module} distributed.fsdp2
```

```{autodoc2-docstring} distributed.fsdp2
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FSDP2Manager <distributed.fsdp2.FSDP2Manager>`
  - ```{autodoc2-docstring} distributed.fsdp2.FSDP2Manager
    :summary:
    ```
````

### API

`````{py:class} FSDP2Manager
:canonical: distributed.fsdp2.FSDP2Manager

```{autodoc2-docstring} distributed.fsdp2.FSDP2Manager
```

````{py:attribute} dp_size
:canonical: distributed.fsdp2.FSDP2Manager.dp_size
:type: typing.Optional[int]
:value: >
   'field(...)'

```{autodoc2-docstring} distributed.fsdp2.FSDP2Manager.dp_size
```

````

````{py:attribute} dp_replicate_size
:canonical: distributed.fsdp2.FSDP2Manager.dp_replicate_size
:type: typing.Optional[int]
:value: >
   'field(...)'

```{autodoc2-docstring} distributed.fsdp2.FSDP2Manager.dp_replicate_size
```

````

````{py:attribute} tp_size
:canonical: distributed.fsdp2.FSDP2Manager.tp_size
:type: typing.Optional[int]
:value: >
   'field(...)'

```{autodoc2-docstring} distributed.fsdp2.FSDP2Manager.tp_size
```

````

````{py:attribute} cp_size
:canonical: distributed.fsdp2.FSDP2Manager.cp_size
:type: typing.Optional[int]
:value: >
   'field(...)'

```{autodoc2-docstring} distributed.fsdp2.FSDP2Manager.cp_size
```

````

````{py:attribute} sequence_parallel
:canonical: distributed.fsdp2.FSDP2Manager.sequence_parallel
:type: typing.Optional[bool]
:value: >
   'field(...)'

```{autodoc2-docstring} distributed.fsdp2.FSDP2Manager.sequence_parallel
```

````

````{py:attribute} mp_policy
:canonical: distributed.fsdp2.FSDP2Manager.mp_policy
:type: typing.Optional[torch.distributed.fsdp.MixedPrecisionPolicy]
:value: >
   'field(...)'

```{autodoc2-docstring} distributed.fsdp2.FSDP2Manager.mp_policy
```

````

````{py:attribute} offload_policy
:canonical: distributed.fsdp2.FSDP2Manager.offload_policy
:type: typing.Optional[torch.distributed.fsdp.CPUOffloadPolicy]
:value: >
   'field(...)'

```{autodoc2-docstring} distributed.fsdp2.FSDP2Manager.offload_policy
```

````

````{py:attribute} backend
:canonical: distributed.fsdp2.FSDP2Manager.backend
:type: typing.Optional[str]
:value: >
   'field(...)'

```{autodoc2-docstring} distributed.fsdp2.FSDP2Manager.backend
```

````

````{py:attribute} world_size
:canonical: distributed.fsdp2.FSDP2Manager.world_size
:type: typing.Optional[int]
:value: >
   'field(...)'

```{autodoc2-docstring} distributed.fsdp2.FSDP2Manager.world_size
```

````

````{py:method} __post_init__()
:canonical: distributed.fsdp2.FSDP2Manager.__post_init__

```{autodoc2-docstring} distributed.fsdp2.FSDP2Manager.__post_init__
```

````

````{py:method} _setup_distributed()
:canonical: distributed.fsdp2.FSDP2Manager._setup_distributed

```{autodoc2-docstring} distributed.fsdp2.FSDP2Manager._setup_distributed
```

````

````{py:method} _get_device_mesh()
:canonical: distributed.fsdp2.FSDP2Manager._get_device_mesh

```{autodoc2-docstring} distributed.fsdp2.FSDP2Manager._get_device_mesh
```

````

````{py:method} parallelize(model, use_hf_tp_plan=False)
:canonical: distributed.fsdp2.FSDP2Manager.parallelize

```{autodoc2-docstring} distributed.fsdp2.FSDP2Manager.parallelize
```

````

`````
