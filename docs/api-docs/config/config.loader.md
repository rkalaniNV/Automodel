# {py:mod}`config.loader`

```{py:module} config.loader
```

```{autodoc2-docstring} config.loader
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ConfigNode <config.loader.ConfigNode>`
  - ```{autodoc2-docstring} config.loader.ConfigNode
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`translate_value <config.loader.translate_value>`
  - ```{autodoc2-docstring} config.loader.translate_value
    :summary:
    ```
* - {py:obj}`load_module_from_file <config.loader.load_module_from_file>`
  - ```{autodoc2-docstring} config.loader.load_module_from_file
    :summary:
    ```
* - {py:obj}`_resolve_target <config.loader._resolve_target>`
  - ```{autodoc2-docstring} config.loader._resolve_target
    :summary:
    ```
* - {py:obj}`load_yaml_config <config.loader.load_yaml_config>`
  - ```{autodoc2-docstring} config.loader.load_yaml_config
    :summary:
    ```
````

### API

````{py:function} translate_value(v)
:canonical: config.loader.translate_value

```{autodoc2-docstring} config.loader.translate_value
```
````

````{py:function} load_module_from_file(file_path)
:canonical: config.loader.load_module_from_file

```{autodoc2-docstring} config.loader.load_module_from_file
```
````

````{py:function} _resolve_target(dotted_path: str)
:canonical: config.loader._resolve_target

```{autodoc2-docstring} config.loader._resolve_target
```
````

`````{py:class} ConfigNode(d, raise_on_missing_attr=True)
:canonical: config.loader.ConfigNode

```{autodoc2-docstring} config.loader.ConfigNode
```

```{rubric} Initialization
```

```{autodoc2-docstring} config.loader.ConfigNode.__init__
```

````{py:method} __getattr__(key)
:canonical: config.loader.ConfigNode.__getattr__

```{autodoc2-docstring} config.loader.ConfigNode.__getattr__
```

````

````{py:method} _wrap(k, v)
:canonical: config.loader.ConfigNode._wrap

```{autodoc2-docstring} config.loader.ConfigNode._wrap
```

````

````{py:method} instantiate(*args, **kwargs)
:canonical: config.loader.ConfigNode.instantiate

```{autodoc2-docstring} config.loader.ConfigNode.instantiate
```

````

````{py:method} _instantiate_value(v)
:canonical: config.loader.ConfigNode._instantiate_value

```{autodoc2-docstring} config.loader.ConfigNode._instantiate_value
```

````

````{py:method} to_dict()
:canonical: config.loader.ConfigNode.to_dict

```{autodoc2-docstring} config.loader.ConfigNode.to_dict
```

````

````{py:method} _unwrap(v)
:canonical: config.loader.ConfigNode._unwrap

```{autodoc2-docstring} config.loader.ConfigNode._unwrap
```

````

````{py:method} get(key, default=None)
:canonical: config.loader.ConfigNode.get

```{autodoc2-docstring} config.loader.ConfigNode.get
```

````

````{py:method} set_by_dotted(dotted_key: str, value)
:canonical: config.loader.ConfigNode.set_by_dotted

```{autodoc2-docstring} config.loader.ConfigNode.set_by_dotted
```

````

````{py:method} __repr__(level=0)
:canonical: config.loader.ConfigNode.__repr__

```{autodoc2-docstring} config.loader.ConfigNode.__repr__
```

````

````{py:method} _repr_value(value, level)
:canonical: config.loader.ConfigNode._repr_value

```{autodoc2-docstring} config.loader.ConfigNode._repr_value
```

````

````{py:method} __str__()
:canonical: config.loader.ConfigNode.__str__

```{autodoc2-docstring} config.loader.ConfigNode.__str__
```

````

````{py:method} __contains__(key)
:canonical: config.loader.ConfigNode.__contains__

```{autodoc2-docstring} config.loader.ConfigNode.__contains__
```

````

`````

````{py:function} load_yaml_config(path)
:canonical: config.loader.load_yaml_config

```{autodoc2-docstring} config.loader.load_yaml_config
```
````
