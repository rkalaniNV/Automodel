# {py:mod}`nemo_automodel.components.training.rng`

```{py:module} nemo_automodel.components.training.rng
```

```{autodoc2-docstring} nemo_automodel.components.training.rng
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`StatefulRNG <nemo_automodel.components.training.rng.StatefulRNG>`
  - ```{autodoc2-docstring} nemo_automodel.components.training.rng.StatefulRNG
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`init_all_rng <nemo_automodel.components.training.rng.init_all_rng>`
  - ```{autodoc2-docstring} nemo_automodel.components.training.rng.init_all_rng
    :summary:
    ```
````

### API

````{py:function} init_all_rng(seed: int, ranked: bool = False)
:canonical: nemo_automodel.components.training.rng.init_all_rng

```{autodoc2-docstring} nemo_automodel.components.training.rng.init_all_rng
```
````

`````{py:class} StatefulRNG(seed: int, ranked: bool = False)
:canonical: nemo_automodel.components.training.rng.StatefulRNG

```{autodoc2-docstring} nemo_automodel.components.training.rng.StatefulRNG
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_automodel.components.training.rng.StatefulRNG.__init__
```

````{py:method} __del__()
:canonical: nemo_automodel.components.training.rng.StatefulRNG.__del__

```{autodoc2-docstring} nemo_automodel.components.training.rng.StatefulRNG.__del__
```

````

````{py:method} state_dict()
:canonical: nemo_automodel.components.training.rng.StatefulRNG.state_dict

```{autodoc2-docstring} nemo_automodel.components.training.rng.StatefulRNG.state_dict
```

````

````{py:method} load_state_dict(state)
:canonical: nemo_automodel.components.training.rng.StatefulRNG.load_state_dict

```{autodoc2-docstring} nemo_automodel.components.training.rng.StatefulRNG.load_state_dict
```

````

````{py:method} __enter__()
:canonical: nemo_automodel.components.training.rng.StatefulRNG.__enter__

```{autodoc2-docstring} nemo_automodel.components.training.rng.StatefulRNG.__enter__
```

````

````{py:method} __exit__(exc_type, exc_value, traceback)
:canonical: nemo_automodel.components.training.rng.StatefulRNG.__exit__

```{autodoc2-docstring} nemo_automodel.components.training.rng.StatefulRNG.__exit__
```

````

`````
