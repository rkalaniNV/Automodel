# {py:mod}`loss.masked_ce`

```{py:module} loss.masked_ce
```

```{autodoc2-docstring} loss.masked_ce
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MaskedCrossEntropy <loss.masked_ce.MaskedCrossEntropy>`
  - ```{autodoc2-docstring} loss.masked_ce.MaskedCrossEntropy
    :summary:
    ```
````

### API

`````{py:class} MaskedCrossEntropy(fp32_upcast: bool = True, ignore_index: int = -100, reduction: str = 'sum')
:canonical: loss.masked_ce.MaskedCrossEntropy

```{autodoc2-docstring} loss.masked_ce.MaskedCrossEntropy
```

```{rubric} Initialization
```

```{autodoc2-docstring} loss.masked_ce.MaskedCrossEntropy.__init__
```

````{py:method} __call__(logits: torch.Tensor, labels: torch.Tensor, mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: loss.masked_ce.MaskedCrossEntropy.__call__

```{autodoc2-docstring} loss.masked_ce.MaskedCrossEntropy.__call__
```

````

`````
