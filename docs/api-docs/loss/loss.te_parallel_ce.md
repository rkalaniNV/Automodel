# {py:mod}`loss.te_parallel_ce`

```{py:module} loss.te_parallel_ce
```

```{autodoc2-docstring} loss.te_parallel_ce
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CrossEntropyFunction <loss.te_parallel_ce.CrossEntropyFunction>`
  - ```{autodoc2-docstring} loss.te_parallel_ce.CrossEntropyFunction
    :summary:
    ```
* - {py:obj}`TEParallelCrossEntropy <loss.te_parallel_ce.TEParallelCrossEntropy>`
  - ```{autodoc2-docstring} loss.te_parallel_ce.TEParallelCrossEntropy
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HAVE_TE_PARALLEL_CE <loss.te_parallel_ce.HAVE_TE_PARALLEL_CE>`
  - ```{autodoc2-docstring} loss.te_parallel_ce.HAVE_TE_PARALLEL_CE
    :summary:
    ```
* - {py:obj}`MISSING_TE_PARALLEL_CE_MSG <loss.te_parallel_ce.MISSING_TE_PARALLEL_CE_MSG>`
  - ```{autodoc2-docstring} loss.te_parallel_ce.MISSING_TE_PARALLEL_CE_MSG
    :summary:
    ```
* - {py:obj}`parallel_cross_entropy <loss.te_parallel_ce.parallel_cross_entropy>`
  - ```{autodoc2-docstring} loss.te_parallel_ce.parallel_cross_entropy
    :summary:
    ```
````

### API

````{py:data} HAVE_TE_PARALLEL_CE
:canonical: loss.te_parallel_ce.HAVE_TE_PARALLEL_CE
:value: >
   None

```{autodoc2-docstring} loss.te_parallel_ce.HAVE_TE_PARALLEL_CE
```

````

````{py:data} MISSING_TE_PARALLEL_CE_MSG
:canonical: loss.te_parallel_ce.MISSING_TE_PARALLEL_CE_MSG
:value: >
   None

```{autodoc2-docstring} loss.te_parallel_ce.MISSING_TE_PARALLEL_CE_MSG
```

````

`````{py:class} CrossEntropyFunction(*args, **kwargs)
:canonical: loss.te_parallel_ce.CrossEntropyFunction

Bases: {py:obj}`torch.autograd.Function`

```{autodoc2-docstring} loss.te_parallel_ce.CrossEntropyFunction
```

```{rubric} Initialization
```

```{autodoc2-docstring} loss.te_parallel_ce.CrossEntropyFunction.__init__
```

````{py:method} forward(ctx, _input, target, label_smoothing=0.0, reduce_loss=False, dist_process_group=None, ignore_idx=-100)
:canonical: loss.te_parallel_ce.CrossEntropyFunction.forward
:staticmethod:

```{autodoc2-docstring} loss.te_parallel_ce.CrossEntropyFunction.forward
```

````

````{py:method} backward(ctx, grad_output)
:canonical: loss.te_parallel_ce.CrossEntropyFunction.backward
:staticmethod:

```{autodoc2-docstring} loss.te_parallel_ce.CrossEntropyFunction.backward
```

````

`````

````{py:data} parallel_cross_entropy
:canonical: loss.te_parallel_ce.parallel_cross_entropy
:value: >
   None

```{autodoc2-docstring} loss.te_parallel_ce.parallel_cross_entropy
```

````

`````{py:class} TEParallelCrossEntropy(ignore_index: int = -100, reduction: str = 'sum', tp_group: typing.Optional[torch.distributed.ProcessGroup] = None)
:canonical: loss.te_parallel_ce.TEParallelCrossEntropy

```{autodoc2-docstring} loss.te_parallel_ce.TEParallelCrossEntropy
```

```{rubric} Initialization
```

```{autodoc2-docstring} loss.te_parallel_ce.TEParallelCrossEntropy.__init__
```

````{py:method} __call__(logits: torch.Tensor, labels: torch.Tensor, mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: loss.te_parallel_ce.TEParallelCrossEntropy.__call__

```{autodoc2-docstring} loss.te_parallel_ce.TEParallelCrossEntropy.__call__
```

````

`````
