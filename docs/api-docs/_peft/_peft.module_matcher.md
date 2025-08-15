# {py:mod}`_peft.module_matcher`

```{py:module} _peft.module_matcher
```

```{autodoc2-docstring} _peft.module_matcher
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ModuleMatcher <_peft.module_matcher.ModuleMatcher>`
  - ```{autodoc2-docstring} _peft.module_matcher.ModuleMatcher
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`wildcard_match <_peft.module_matcher.wildcard_match>`
  - ```{autodoc2-docstring} _peft.module_matcher.wildcard_match
    :summary:
    ```
````

### API

````{py:function} wildcard_match(pattern, key)
:canonical: _peft.module_matcher.wildcard_match

```{autodoc2-docstring} _peft.module_matcher.wildcard_match
```
````

`````{py:class} ModuleMatcher
:canonical: _peft.module_matcher.ModuleMatcher

```{autodoc2-docstring} _peft.module_matcher.ModuleMatcher
```

````{py:attribute} target_modules
:canonical: _peft.module_matcher.ModuleMatcher.target_modules
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} _peft.module_matcher.ModuleMatcher.target_modules
```

````

````{py:attribute} exclude_modules
:canonical: _peft.module_matcher.ModuleMatcher.exclude_modules
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} _peft.module_matcher.ModuleMatcher.exclude_modules
```

````

````{py:attribute} match_all_linear
:canonical: _peft.module_matcher.ModuleMatcher.match_all_linear
:type: bool
:value: >
   'field(...)'

```{autodoc2-docstring} _peft.module_matcher.ModuleMatcher.match_all_linear
```

````

````{py:attribute} is_causal_lm
:canonical: _peft.module_matcher.ModuleMatcher.is_causal_lm
:type: bool
:value: >
   'field(...)'

```{autodoc2-docstring} _peft.module_matcher.ModuleMatcher.is_causal_lm
```

````

````{py:method} __post_init__()
:canonical: _peft.module_matcher.ModuleMatcher.__post_init__

```{autodoc2-docstring} _peft.module_matcher.ModuleMatcher.__post_init__
```

````

````{py:method} match(m: torch.nn.Module, name: str = None, prefix: str = None)
:canonical: _peft.module_matcher.ModuleMatcher.match

```{autodoc2-docstring} _peft.module_matcher.ModuleMatcher.match
```

````

`````
