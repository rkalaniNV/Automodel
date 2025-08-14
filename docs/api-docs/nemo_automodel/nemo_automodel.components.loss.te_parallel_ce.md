# {py:mod}`nemo_automodel.components.loss.te_parallel_ce`

```{py:module} nemo_automodel.components.loss.te_parallel_ce
```

```{autodoc2-docstring} nemo_automodel.components.loss.te_parallel_ce
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CrossEntropyFunction <nemo_automodel.components.loss.te_parallel_ce.CrossEntropyFunction>`
  - ```{autodoc2-docstring} nemo_automodel.components.loss.te_parallel_ce.CrossEntropyFunction
    :summary:
    ```
* - {py:obj}`TEParallelCrossEntropy <nemo_automodel.components.loss.te_parallel_ce.TEParallelCrossEntropy>`
  - ```{autodoc2-docstring} nemo_automodel.components.loss.te_parallel_ce.TEParallelCrossEntropy
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HAVE_TE_PARALLEL_CE <nemo_automodel.components.loss.te_parallel_ce.HAVE_TE_PARALLEL_CE>`
  - ```{autodoc2-docstring} nemo_automodel.components.loss.te_parallel_ce.HAVE_TE_PARALLEL_CE
    :summary:
    ```
* - {py:obj}`MISSING_TE_PARALLEL_CE_MSG <nemo_automodel.components.loss.te_parallel_ce.MISSING_TE_PARALLEL_CE_MSG>`
  - ```{autodoc2-docstring} nemo_automodel.components.loss.te_parallel_ce.MISSING_TE_PARALLEL_CE_MSG
    :summary:
    ```
* - {py:obj}`parallel_cross_entropy <nemo_automodel.components.loss.te_parallel_ce.parallel_cross_entropy>`
  - ```{autodoc2-docstring} nemo_automodel.components.loss.te_parallel_ce.parallel_cross_entropy
    :summary:
    ```
````

### API

````{py:data} HAVE_TE_PARALLEL_CE
:canonical: nemo_automodel.components.loss.te_parallel_ce.HAVE_TE_PARALLEL_CE
:value: >
   None

```{autodoc2-docstring} nemo_automodel.components.loss.te_parallel_ce.HAVE_TE_PARALLEL_CE
```

````

````{py:data} MISSING_TE_PARALLEL_CE_MSG
:canonical: nemo_automodel.components.loss.te_parallel_ce.MISSING_TE_PARALLEL_CE_MSG
:value: >
   None

```{autodoc2-docstring} nemo_automodel.components.loss.te_parallel_ce.MISSING_TE_PARALLEL_CE_MSG
```

````

`````{py:class} CrossEntropyFunction
:canonical: nemo_automodel.components.loss.te_parallel_ce.CrossEntropyFunction

Bases: {py:obj}`torch.autograd.Function`

```{autodoc2-docstring} nemo_automodel.components.loss.te_parallel_ce.CrossEntropyFunction
```

````{py:method} forward(ctx, _input, target, label_smoothing=0.0, reduce_loss=False, dist_process_group=None, ignore_idx=-100)
:canonical: nemo_automodel.components.loss.te_parallel_ce.CrossEntropyFunction.forward
:staticmethod:

```{autodoc2-docstring} nemo_automodel.components.loss.te_parallel_ce.CrossEntropyFunction.forward
```

````

````{py:method} backward(ctx, grad_output)
:canonical: nemo_automodel.components.loss.te_parallel_ce.CrossEntropyFunction.backward
:staticmethod:

```{autodoc2-docstring} nemo_automodel.components.loss.te_parallel_ce.CrossEntropyFunction.backward
```

````

`````

````{py:data} parallel_cross_entropy
:canonical: nemo_automodel.components.loss.te_parallel_ce.parallel_cross_entropy
:value: >
   None

```{autodoc2-docstring} nemo_automodel.components.loss.te_parallel_ce.parallel_cross_entropy
```

````

`````{py:class} TEParallelCrossEntropy(ignore_index: int = -100, reduction: str = 'sum', tp_group: typing.Optional[torch.distributed.ProcessGroup] = None)
:canonical: nemo_automodel.components.loss.te_parallel_ce.TEParallelCrossEntropy

```{autodoc2-docstring} nemo_automodel.components.loss.te_parallel_ce.TEParallelCrossEntropy
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_automodel.components.loss.te_parallel_ce.TEParallelCrossEntropy.__init__
```

````{py:method} __call__(logits: torch.Tensor, labels: torch.Tensor, mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: nemo_automodel.components.loss.te_parallel_ce.TEParallelCrossEntropy.__call__

```{autodoc2-docstring} nemo_automodel.components.loss.te_parallel_ce.TEParallelCrossEntropy.__call__
```

````

`````
