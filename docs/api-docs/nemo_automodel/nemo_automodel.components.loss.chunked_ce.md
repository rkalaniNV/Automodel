# {py:mod}`nemo_automodel.components.loss.chunked_ce`

```{py:module} nemo_automodel.components.loss.chunked_ce
```

```{autodoc2-docstring} nemo_automodel.components.loss.chunked_ce
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ChunkedCrossEntropy <nemo_automodel.components.loss.chunked_ce.ChunkedCrossEntropy>`
  - ```{autodoc2-docstring} nemo_automodel.components.loss.chunked_ce.ChunkedCrossEntropy
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`compute_cross_entropy <nemo_automodel.components.loss.chunked_ce.compute_cross_entropy>`
  - ```{autodoc2-docstring} nemo_automodel.components.loss.chunked_ce.compute_cross_entropy
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_compiled_compute_cross_entropy <nemo_automodel.components.loss.chunked_ce._compiled_compute_cross_entropy>`
  - ```{autodoc2-docstring} nemo_automodel.components.loss.chunked_ce._compiled_compute_cross_entropy
    :summary:
    ```
````

### API

````{py:data} _compiled_compute_cross_entropy
:canonical: nemo_automodel.components.loss.chunked_ce._compiled_compute_cross_entropy
:value: >
   None

```{autodoc2-docstring} nemo_automodel.components.loss.chunked_ce._compiled_compute_cross_entropy
```

````

````{py:function} compute_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index=-100)
:canonical: nemo_automodel.components.loss.chunked_ce.compute_cross_entropy

```{autodoc2-docstring} nemo_automodel.components.loss.chunked_ce.compute_cross_entropy
```
````

`````{py:class} ChunkedCrossEntropy(chunk_len: int = 32, compile: bool = True, ignore_index: int = -100)
:canonical: nemo_automodel.components.loss.chunked_ce.ChunkedCrossEntropy

```{autodoc2-docstring} nemo_automodel.components.loss.chunked_ce.ChunkedCrossEntropy
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_automodel.components.loss.chunked_ce.ChunkedCrossEntropy.__init__
```

````{py:method} __call__(logits: torch.Tensor, labels: torch.Tensor, mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: nemo_automodel.components.loss.chunked_ce.ChunkedCrossEntropy.__call__

```{autodoc2-docstring} nemo_automodel.components.loss.chunked_ce.ChunkedCrossEntropy.__call__
```

````

`````
