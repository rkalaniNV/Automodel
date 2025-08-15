# {py:mod}`loss.linear_ce`

```{py:module} loss.linear_ce
```

```{autodoc2-docstring} loss.linear_ce
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FusedLinearCrossEntropy <loss.linear_ce.FusedLinearCrossEntropy>`
  - ```{autodoc2-docstring} loss.linear_ce.FusedLinearCrossEntropy
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`new_is_triton_greater_or_equal <loss.linear_ce.new_is_triton_greater_or_equal>`
  - ```{autodoc2-docstring} loss.linear_ce.new_is_triton_greater_or_equal
    :summary:
    ```
* - {py:obj}`new_is_triton_greater_or_equal_3_2_0 <loss.linear_ce.new_is_triton_greater_or_equal_3_2_0>`
  - ```{autodoc2-docstring} loss.linear_ce.new_is_triton_greater_or_equal_3_2_0
    :summary:
    ```
````

### API

````{py:function} new_is_triton_greater_or_equal(version_str)
:canonical: loss.linear_ce.new_is_triton_greater_or_equal

```{autodoc2-docstring} loss.linear_ce.new_is_triton_greater_or_equal
```
````

````{py:function} new_is_triton_greater_or_equal_3_2_0()
:canonical: loss.linear_ce.new_is_triton_greater_or_equal_3_2_0

```{autodoc2-docstring} loss.linear_ce.new_is_triton_greater_or_equal_3_2_0
```
````

`````{py:class} FusedLinearCrossEntropy(ignore_index: int = -100, logit_softcapping: float = 0, reduction: str = 'sum')
:canonical: loss.linear_ce.FusedLinearCrossEntropy

```{autodoc2-docstring} loss.linear_ce.FusedLinearCrossEntropy
```

```{rubric} Initialization
```

```{autodoc2-docstring} loss.linear_ce.FusedLinearCrossEntropy.__init__
```

````{py:method} __call__(hidden_states: torch.Tensor, labels: torch.Tensor, lm_weight: torch.Tensor) -> torch.Tensor
:canonical: loss.linear_ce.FusedLinearCrossEntropy.__call__

```{autodoc2-docstring} loss.linear_ce.FusedLinearCrossEntropy.__call__
```

````

`````
