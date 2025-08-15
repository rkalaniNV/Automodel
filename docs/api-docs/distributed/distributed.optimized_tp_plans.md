# {py:mod}`distributed.optimized_tp_plans`

```{py:module} distributed.optimized_tp_plans
```

```{autodoc2-docstring} distributed.optimized_tp_plans
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RotaryEmbedParallel <distributed.optimized_tp_plans.RotaryEmbedParallel>`
  - ```{autodoc2-docstring} distributed.optimized_tp_plans.RotaryEmbedParallel
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_parallelize_gemma3 <distributed.optimized_tp_plans._parallelize_gemma3>`
  - ```{autodoc2-docstring} distributed.optimized_tp_plans._parallelize_gemma3
    :summary:
    ```
* - {py:obj}`_parallelize_llama <distributed.optimized_tp_plans._parallelize_llama>`
  - ```{autodoc2-docstring} distributed.optimized_tp_plans._parallelize_llama
    :summary:
    ```
* - {py:obj}`_parallelize_qwen <distributed.optimized_tp_plans._parallelize_qwen>`
  - ```{autodoc2-docstring} distributed.optimized_tp_plans._parallelize_qwen
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PARALLELIZE_FUNCTIONS <distributed.optimized_tp_plans.PARALLELIZE_FUNCTIONS>`
  - ```{autodoc2-docstring} distributed.optimized_tp_plans.PARALLELIZE_FUNCTIONS
    :summary:
    ```
````

### API

`````{py:class} RotaryEmbedParallel(*, sequence_dim: int = 1, use_local_output: bool = False)
:canonical: distributed.optimized_tp_plans.RotaryEmbedParallel

Bases: {py:obj}`torch.distributed.tensor.parallel.SequenceParallel`

```{autodoc2-docstring} distributed.optimized_tp_plans.RotaryEmbedParallel
```

```{rubric} Initialization
```

```{autodoc2-docstring} distributed.optimized_tp_plans.RotaryEmbedParallel.__init__
```

````{py:method} _prepare_input_fn(sequence_sharding, mod, inputs, device_mesh)
:canonical: distributed.optimized_tp_plans.RotaryEmbedParallel._prepare_input_fn
:staticmethod:

```{autodoc2-docstring} distributed.optimized_tp_plans.RotaryEmbedParallel._prepare_input_fn
```

````

````{py:method} _prepare_output_fn(use_local_output, mod, outputs, device_mesh)
:canonical: distributed.optimized_tp_plans.RotaryEmbedParallel._prepare_output_fn
:staticmethod:

```{autodoc2-docstring} distributed.optimized_tp_plans.RotaryEmbedParallel._prepare_output_fn
```

````

`````

````{py:function} _parallelize_gemma3(model: typing.Union[transformers.models.gemma3.modeling_gemma3.Gemma3ForCausalLM, transformers.models.gemma3.modeling_gemma3.Gemma3ForConditionalGeneration], sequence_parallel: bool = False)
:canonical: distributed.optimized_tp_plans._parallelize_gemma3

```{autodoc2-docstring} distributed.optimized_tp_plans._parallelize_gemma3
```
````

````{py:function} _parallelize_llama(model: transformers.models.llama.modeling_llama.LlamaForCausalLM, sequence_parallel: bool = False)
:canonical: distributed.optimized_tp_plans._parallelize_llama

```{autodoc2-docstring} distributed.optimized_tp_plans._parallelize_llama
```
````

````{py:function} _parallelize_qwen(model: typing.Union[transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM, transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM], sequence_parallel: bool = False)
:canonical: distributed.optimized_tp_plans._parallelize_qwen

```{autodoc2-docstring} distributed.optimized_tp_plans._parallelize_qwen
```
````

````{py:data} PARALLELIZE_FUNCTIONS
:canonical: distributed.optimized_tp_plans.PARALLELIZE_FUNCTIONS
:type: typing.Dict[type, typing.Callable[..., typing.Dict[str, torch.distributed.tensor.parallel.ParallelStyle]]]
:value: >
   None

```{autodoc2-docstring} distributed.optimized_tp_plans.PARALLELIZE_FUNCTIONS
```

````
