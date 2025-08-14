# {py:mod}`nemo_automodel.components._peft.lora`

```{py:module} nemo_automodel.components._peft.lora
```

```{autodoc2-docstring} nemo_automodel.components._peft.lora
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PeftConfig <nemo_automodel.components._peft.lora.PeftConfig>`
  - ```{autodoc2-docstring} nemo_automodel.components._peft.lora.PeftConfig
    :summary:
    ```
* - {py:obj}`LinearLoRA <nemo_automodel.components._peft.lora.LinearLoRA>`
  - ```{autodoc2-docstring} nemo_automodel.components._peft.lora.LinearLoRA
    :summary:
    ```
* - {py:obj}`TritonLinearLoRA <nemo_automodel.components._peft.lora.TritonLinearLoRA>`
  - ```{autodoc2-docstring} nemo_automodel.components._peft.lora.TritonLinearLoRA
    :summary:
    ```
* - {py:obj}`LoRATritonFunction <nemo_automodel.components._peft.lora.LoRATritonFunction>`
  - ```{autodoc2-docstring} nemo_automodel.components._peft.lora.LoRATritonFunction
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`patch_linear_module <nemo_automodel.components._peft.lora.patch_linear_module>`
  - ```{autodoc2-docstring} nemo_automodel.components._peft.lora.patch_linear_module
    :summary:
    ```
* - {py:obj}`apply_lora_to_linear_modules <nemo_automodel.components._peft.lora.apply_lora_to_linear_modules>`
  - ```{autodoc2-docstring} nemo_automodel.components._peft.lora.apply_lora_to_linear_modules
    :summary:
    ```
````

### API

`````{py:class} PeftConfig
:canonical: nemo_automodel.components._peft.lora.PeftConfig

```{autodoc2-docstring} nemo_automodel.components._peft.lora.PeftConfig
```

````{py:attribute} target_modules
:canonical: nemo_automodel.components._peft.lora.PeftConfig.target_modules
:type: list
:value: >
   'field(...)'

```{autodoc2-docstring} nemo_automodel.components._peft.lora.PeftConfig.target_modules
```

````

````{py:attribute} exclude_modules
:canonical: nemo_automodel.components._peft.lora.PeftConfig.exclude_modules
:type: list
:value: >
   'field(...)'

```{autodoc2-docstring} nemo_automodel.components._peft.lora.PeftConfig.exclude_modules
```

````

````{py:attribute} match_all_linear
:canonical: nemo_automodel.components._peft.lora.PeftConfig.match_all_linear
:type: bool
:value: >
   False

```{autodoc2-docstring} nemo_automodel.components._peft.lora.PeftConfig.match_all_linear
```

````

````{py:attribute} dim
:canonical: nemo_automodel.components._peft.lora.PeftConfig.dim
:type: int
:value: >
   8

```{autodoc2-docstring} nemo_automodel.components._peft.lora.PeftConfig.dim
```

````

````{py:attribute} alpha
:canonical: nemo_automodel.components._peft.lora.PeftConfig.alpha
:type: int
:value: >
   32

```{autodoc2-docstring} nemo_automodel.components._peft.lora.PeftConfig.alpha
```

````

````{py:attribute} dropout
:canonical: nemo_automodel.components._peft.lora.PeftConfig.dropout
:type: float
:value: >
   0.0

```{autodoc2-docstring} nemo_automodel.components._peft.lora.PeftConfig.dropout
```

````

````{py:attribute} dropout_position
:canonical: nemo_automodel.components._peft.lora.PeftConfig.dropout_position
:type: typing.Literal[pre, post]
:value: >
   'post'

```{autodoc2-docstring} nemo_automodel.components._peft.lora.PeftConfig.dropout_position
```

````

````{py:attribute} lora_A_init
:canonical: nemo_automodel.components._peft.lora.PeftConfig.lora_A_init
:type: str
:value: >
   'xavier'

```{autodoc2-docstring} nemo_automodel.components._peft.lora.PeftConfig.lora_A_init
```

````

````{py:attribute} lora_dtype
:canonical: nemo_automodel.components._peft.lora.PeftConfig.lora_dtype
:type: typing.Optional[torch.dtype]
:value: >
   None

```{autodoc2-docstring} nemo_automodel.components._peft.lora.PeftConfig.lora_dtype
```

````

````{py:attribute} use_triton
:canonical: nemo_automodel.components._peft.lora.PeftConfig.use_triton
:type: bool
:value: >
   False

```{autodoc2-docstring} nemo_automodel.components._peft.lora.PeftConfig.use_triton
```

````

````{py:method} to_dict()
:canonical: nemo_automodel.components._peft.lora.PeftConfig.to_dict

```{autodoc2-docstring} nemo_automodel.components._peft.lora.PeftConfig.to_dict
```

````

````{py:method} from_dict(d: dict[str, typing.Any])
:canonical: nemo_automodel.components._peft.lora.PeftConfig.from_dict
:classmethod:

```{autodoc2-docstring} nemo_automodel.components._peft.lora.PeftConfig.from_dict
```

````

`````

`````{py:class} LinearLoRA(orig_linear, dim=8, alpha=32, dropout=0.0, dropout_position='post', lora_A_init_method='xavier', lora_dtype=None)
:canonical: nemo_automodel.components._peft.lora.LinearLoRA

Bases: {py:obj}`torch.nn.Linear`

```{autodoc2-docstring} nemo_automodel.components._peft.lora.LinearLoRA
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_automodel.components._peft.lora.LinearLoRA.__init__
```

````{py:method} _init_adapter(obj, dim=8, alpha=32, dropout=0.0, dropout_position='post', lora_A_init_method='xavier', lora_dtype=None)
:canonical: nemo_automodel.components._peft.lora.LinearLoRA._init_adapter
:staticmethod:

```{autodoc2-docstring} nemo_automodel.components._peft.lora.LinearLoRA._init_adapter
```

````

````{py:method} forward(x)
:canonical: nemo_automodel.components._peft.lora.LinearLoRA.forward

```{autodoc2-docstring} nemo_automodel.components._peft.lora.LinearLoRA.forward
```

````

`````

`````{py:class} TritonLinearLoRA(orig_linear, dim=8, alpha=32, dropout=0.0, dropout_position='post', lora_A_init_method='xavier', lora_dtype=None)
:canonical: nemo_automodel.components._peft.lora.TritonLinearLoRA

Bases: {py:obj}`nemo_automodel.components._peft.lora.LinearLoRA`

```{autodoc2-docstring} nemo_automodel.components._peft.lora.TritonLinearLoRA
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_automodel.components._peft.lora.TritonLinearLoRA.__init__
```

````{py:method} forward(x)
:canonical: nemo_automodel.components._peft.lora.TritonLinearLoRA.forward

```{autodoc2-docstring} nemo_automodel.components._peft.lora.TritonLinearLoRA.forward
```

````

`````

````{py:function} patch_linear_module(orig_linear, dim=8, alpha=32, dropout=0.0, dropout_position='post', lora_A_init_method='xavier', lora_dtype=None, use_triton=True)
:canonical: nemo_automodel.components._peft.lora.patch_linear_module

```{autodoc2-docstring} nemo_automodel.components._peft.lora.patch_linear_module
```
````

````{py:function} apply_lora_to_linear_modules(model: torch.nn.Module, peft_config: nemo_automodel.components._peft.lora.PeftConfig) -> int
:canonical: nemo_automodel.components._peft.lora.apply_lora_to_linear_modules

```{autodoc2-docstring} nemo_automodel.components._peft.lora.apply_lora_to_linear_modules
```
````

`````{py:class} LoRATritonFunction
:canonical: nemo_automodel.components._peft.lora.LoRATritonFunction

Bases: {py:obj}`torch.autograd.Function`

```{autodoc2-docstring} nemo_automodel.components._peft.lora.LoRATritonFunction
```

````{py:method} setup_context(ctx, inputs, output)
:canonical: nemo_automodel.components._peft.lora.LoRATritonFunction.setup_context
:staticmethod:

```{autodoc2-docstring} nemo_automodel.components._peft.lora.LoRATritonFunction.setup_context
```

````

````{py:method} forward(x, lora_A, lora_B, scale, dtype)
:canonical: nemo_automodel.components._peft.lora.LoRATritonFunction.forward
:staticmethod:

```{autodoc2-docstring} nemo_automodel.components._peft.lora.LoRATritonFunction.forward
```

````

````{py:method} backward(ctx, d_y)
:canonical: nemo_automodel.components._peft.lora.LoRATritonFunction.backward
:staticmethod:

```{autodoc2-docstring} nemo_automodel.components._peft.lora.LoRATritonFunction.backward
```

````

`````
