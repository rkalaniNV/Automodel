# {py:mod}`nemo_automodel.shared.import_utils`

```{py:module} nemo_automodel.shared.import_utils
```

```{autodoc2-docstring} nemo_automodel.shared.import_utils
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`UnavailableMeta <nemo_automodel.shared.import_utils.UnavailableMeta>`
  - ```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`null_decorator <nemo_automodel.shared.import_utils.null_decorator>`
  - ```{autodoc2-docstring} nemo_automodel.shared.import_utils.null_decorator
    :summary:
    ```
* - {py:obj}`is_unavailable <nemo_automodel.shared.import_utils.is_unavailable>`
  - ```{autodoc2-docstring} nemo_automodel.shared.import_utils.is_unavailable
    :summary:
    ```
* - {py:obj}`safe_import <nemo_automodel.shared.import_utils.safe_import>`
  - ```{autodoc2-docstring} nemo_automodel.shared.import_utils.safe_import
    :summary:
    ```
* - {py:obj}`safe_import_from <nemo_automodel.shared.import_utils.safe_import_from>`
  - ```{autodoc2-docstring} nemo_automodel.shared.import_utils.safe_import_from
    :summary:
    ```
* - {py:obj}`gpu_only_import <nemo_automodel.shared.import_utils.gpu_only_import>`
  - ```{autodoc2-docstring} nemo_automodel.shared.import_utils.gpu_only_import
    :summary:
    ```
* - {py:obj}`gpu_only_import_from <nemo_automodel.shared.import_utils.gpu_only_import_from>`
  - ```{autodoc2-docstring} nemo_automodel.shared.import_utils.gpu_only_import_from
    :summary:
    ```
* - {py:obj}`get_torch_version <nemo_automodel.shared.import_utils.get_torch_version>`
  - ```{autodoc2-docstring} nemo_automodel.shared.import_utils.get_torch_version
    :summary:
    ```
* - {py:obj}`is_torch_min_version <nemo_automodel.shared.import_utils.is_torch_min_version>`
  - ```{autodoc2-docstring} nemo_automodel.shared.import_utils.is_torch_min_version
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <nemo_automodel.shared.import_utils.logger>`
  - ```{autodoc2-docstring} nemo_automodel.shared.import_utils.logger
    :summary:
    ```
* - {py:obj}`GPU_INSTALL_STRING <nemo_automodel.shared.import_utils.GPU_INSTALL_STRING>`
  - ```{autodoc2-docstring} nemo_automodel.shared.import_utils.GPU_INSTALL_STRING
    :summary:
    ```
* - {py:obj}`MISSING_TRITON_MSG <nemo_automodel.shared.import_utils.MISSING_TRITON_MSG>`
  - ```{autodoc2-docstring} nemo_automodel.shared.import_utils.MISSING_TRITON_MSG
    :summary:
    ```
* - {py:obj}`MISSING_QWEN_VL_UTILS_MSG <nemo_automodel.shared.import_utils.MISSING_QWEN_VL_UTILS_MSG>`
  - ```{autodoc2-docstring} nemo_automodel.shared.import_utils.MISSING_QWEN_VL_UTILS_MSG
    :summary:
    ```
* - {py:obj}`MISSING_CUT_CROSS_ENTROPY_MSG <nemo_automodel.shared.import_utils.MISSING_CUT_CROSS_ENTROPY_MSG>`
  - ```{autodoc2-docstring} nemo_automodel.shared.import_utils.MISSING_CUT_CROSS_ENTROPY_MSG
    :summary:
    ```
* - {py:obj}`MISSING_TORCHAO_MSG <nemo_automodel.shared.import_utils.MISSING_TORCHAO_MSG>`
  - ```{autodoc2-docstring} nemo_automodel.shared.import_utils.MISSING_TORCHAO_MSG
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: nemo_automodel.shared.import_utils.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} nemo_automodel.shared.import_utils.logger
```

````

````{py:data} GPU_INSTALL_STRING
:canonical: nemo_automodel.shared.import_utils.GPU_INSTALL_STRING
:value: <Multiline-String>

```{autodoc2-docstring} nemo_automodel.shared.import_utils.GPU_INSTALL_STRING
```

````

````{py:data} MISSING_TRITON_MSG
:canonical: nemo_automodel.shared.import_utils.MISSING_TRITON_MSG
:value: >
   'triton is not installed. Please install it with `pip install triton`.'

```{autodoc2-docstring} nemo_automodel.shared.import_utils.MISSING_TRITON_MSG
```

````

````{py:data} MISSING_QWEN_VL_UTILS_MSG
:canonical: nemo_automodel.shared.import_utils.MISSING_QWEN_VL_UTILS_MSG
:value: >
   'qwen_vl_utils is not installed. Please install it with `pip install qwen-vl-utils`.'

```{autodoc2-docstring} nemo_automodel.shared.import_utils.MISSING_QWEN_VL_UTILS_MSG
```

````

````{py:data} MISSING_CUT_CROSS_ENTROPY_MSG
:canonical: nemo_automodel.shared.import_utils.MISSING_CUT_CROSS_ENTROPY_MSG
:value: >
   'cut_cross_entropy is not installed. Please install it with `pip install cut-cross-entropy`.'

```{autodoc2-docstring} nemo_automodel.shared.import_utils.MISSING_CUT_CROSS_ENTROPY_MSG
```

````

````{py:data} MISSING_TORCHAO_MSG
:canonical: nemo_automodel.shared.import_utils.MISSING_TORCHAO_MSG
:value: >
   'torchao is not installed. Please install it with `pip install torchao`.'

```{autodoc2-docstring} nemo_automodel.shared.import_utils.MISSING_TORCHAO_MSG
```

````

````{py:exception} UnavailableError()
:canonical: nemo_automodel.shared.import_utils.UnavailableError

Bases: {py:obj}`Exception`

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableError
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableError.__init__
```

````

````{py:function} null_decorator(*args, **kwargs)
:canonical: nemo_automodel.shared.import_utils.null_decorator

```{autodoc2-docstring} nemo_automodel.shared.import_utils.null_decorator
```
````

`````{py:class} UnavailableMeta
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta

Bases: {py:obj}`type`

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta
```

````{py:method} __new__(name, bases, dct)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__new__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__new__
```

````

````{py:method} __call__(*args, **kwargs)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__call__

````

````{py:method} __getattr__(name)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__getattr__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__getattr__
```

````

````{py:method} __eq__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__eq__

````

````{py:method} __lt__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__lt__

````

````{py:method} __gt__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__gt__

````

````{py:method} __le__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__le__

````

````{py:method} __ge__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__ge__

````

````{py:method} __ne__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__ne__

````

````{py:method} __abs__()
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__abs__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__abs__
```

````

````{py:method} __add__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__add__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__add__
```

````

````{py:method} __radd__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__radd__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__radd__
```

````

````{py:method} __iadd__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__iadd__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__iadd__
```

````

````{py:method} __floordiv__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__floordiv__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__floordiv__
```

````

````{py:method} __rfloordiv__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__rfloordiv__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__rfloordiv__
```

````

````{py:method} __ifloordiv__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__ifloordiv__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__ifloordiv__
```

````

````{py:method} __lshift__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__lshift__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__lshift__
```

````

````{py:method} __rlshift__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__rlshift__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__rlshift__
```

````

````{py:method} __mul__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__mul__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__mul__
```

````

````{py:method} __rmul__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__rmul__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__rmul__
```

````

````{py:method} __imul__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__imul__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__imul__
```

````

````{py:method} __ilshift__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__ilshift__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__ilshift__
```

````

````{py:method} __pow__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__pow__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__pow__
```

````

````{py:method} __rpow__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__rpow__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__rpow__
```

````

````{py:method} __ipow__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__ipow__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__ipow__
```

````

````{py:method} __rshift__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__rshift__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__rshift__
```

````

````{py:method} __rrshift__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__rrshift__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__rrshift__
```

````

````{py:method} __irshift__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__irshift__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__irshift__
```

````

````{py:method} __sub__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__sub__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__sub__
```

````

````{py:method} __rsub__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__rsub__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__rsub__
```

````

````{py:method} __isub__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__isub__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__isub__
```

````

````{py:method} __truediv__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__truediv__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__truediv__
```

````

````{py:method} __rtruediv__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__rtruediv__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__rtruediv__
```

````

````{py:method} __itruediv__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__itruediv__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__itruediv__
```

````

````{py:method} __divmod__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__divmod__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__divmod__
```

````

````{py:method} __rdivmod__(other)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__rdivmod__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__rdivmod__
```

````

````{py:method} __neg__()
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__neg__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__neg__
```

````

````{py:method} __invert__()
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__invert__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__invert__
```

````

````{py:method} __hash__()
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__hash__

````

````{py:method} __index__()
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__index__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__index__
```

````

````{py:method} __iter__()
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__iter__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__iter__
```

````

````{py:method} __delitem__(name)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__delitem__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__delitem__
```

````

````{py:method} __setitem__(name, value)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__setitem__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__setitem__
```

````

````{py:method} __enter__(*args, **kwargs)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__enter__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__enter__
```

````

````{py:method} __get__(*args, **kwargs)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__get__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__get__
```

````

````{py:method} __delete__(*args, **kwargs)
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__delete__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__delete__
```

````

````{py:method} __len__()
:canonical: nemo_automodel.shared.import_utils.UnavailableMeta.__len__

```{autodoc2-docstring} nemo_automodel.shared.import_utils.UnavailableMeta.__len__
```

````

`````

````{py:function} is_unavailable(obj)
:canonical: nemo_automodel.shared.import_utils.is_unavailable

```{autodoc2-docstring} nemo_automodel.shared.import_utils.is_unavailable
```
````

````{py:function} safe_import(module, *, msg=None, alt=None)
:canonical: nemo_automodel.shared.import_utils.safe_import

```{autodoc2-docstring} nemo_automodel.shared.import_utils.safe_import
```
````

````{py:function} safe_import_from(module, symbol, *, msg=None, alt=None, fallback_module=None)
:canonical: nemo_automodel.shared.import_utils.safe_import_from

```{autodoc2-docstring} nemo_automodel.shared.import_utils.safe_import_from
```
````

````{py:function} gpu_only_import(module, *, alt=None)
:canonical: nemo_automodel.shared.import_utils.gpu_only_import

```{autodoc2-docstring} nemo_automodel.shared.import_utils.gpu_only_import
```
````

````{py:function} gpu_only_import_from(module, symbol, *, alt=None)
:canonical: nemo_automodel.shared.import_utils.gpu_only_import_from

```{autodoc2-docstring} nemo_automodel.shared.import_utils.gpu_only_import_from
```
````

````{py:function} get_torch_version()
:canonical: nemo_automodel.shared.import_utils.get_torch_version

```{autodoc2-docstring} nemo_automodel.shared.import_utils.get_torch_version
```
````

````{py:function} is_torch_min_version(version, check_equality=True)
:canonical: nemo_automodel.shared.import_utils.is_torch_min_version

```{autodoc2-docstring} nemo_automodel.shared.import_utils.is_torch_min_version
```
````
