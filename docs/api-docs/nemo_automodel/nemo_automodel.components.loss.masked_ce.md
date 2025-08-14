# {py:mod}`nemo_automodel.components.loss.masked_ce`

```{py:module} nemo_automodel.components.loss.masked_ce
```

```{autodoc2-docstring} nemo_automodel.components.loss.masked_ce
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MaskedCrossEntropy <nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy>`
  - ```{autodoc2-docstring} nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy
    :summary:
    ```
````

### API

`````{py:class} MaskedCrossEntropy(fp32_upcast: bool = True, ignore_index: int = -100, reduction: str = 'sum')
:canonical: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy

```{autodoc2-docstring} nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy.__init__
```

````{py:method} __call__(logits: torch.Tensor, labels: torch.Tensor, mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy.__call__

```{autodoc2-docstring} nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy.__call__
```

````

`````
