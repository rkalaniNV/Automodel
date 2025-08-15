# {py:mod}`optim.scheduler`

```{py:module} optim.scheduler
```

```{autodoc2-docstring} optim.scheduler
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`OptimizerParamScheduler <optim.scheduler.OptimizerParamScheduler>`
  - ```{autodoc2-docstring} optim.scheduler.OptimizerParamScheduler
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <optim.scheduler.logger>`
  - ```{autodoc2-docstring} optim.scheduler.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: optim.scheduler.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} optim.scheduler.logger
```

````

`````{py:class} OptimizerParamScheduler(optimizer: torch.optim.optimizer.Optimizer, init_lr: float, max_lr: float, min_lr: float, lr_warmup_steps: int, lr_decay_steps: int, lr_decay_style: str, start_wd: float, end_wd: float, wd_incr_steps: int, wd_incr_style: str, use_checkpoint_opt_param_scheduler: typing.Optional[bool] = True, override_opt_param_scheduler: typing.Optional[bool] = False, wsd_decay_steps: typing.Optional[int] = None, lr_wsd_decay_style: typing.Optional[str] = None)
:canonical: optim.scheduler.OptimizerParamScheduler

```{autodoc2-docstring} optim.scheduler.OptimizerParamScheduler
```

```{rubric} Initialization
```

```{autodoc2-docstring} optim.scheduler.OptimizerParamScheduler.__init__
```

````{py:method} get_wd() -> float
:canonical: optim.scheduler.OptimizerParamScheduler.get_wd

```{autodoc2-docstring} optim.scheduler.OptimizerParamScheduler.get_wd
```

````

````{py:method} get_lr(param_group: dict) -> float
:canonical: optim.scheduler.OptimizerParamScheduler.get_lr

```{autodoc2-docstring} optim.scheduler.OptimizerParamScheduler.get_lr
```

````

````{py:method} step(increment: int) -> None
:canonical: optim.scheduler.OptimizerParamScheduler.step

```{autodoc2-docstring} optim.scheduler.OptimizerParamScheduler.step
```

````

````{py:method} state_dict() -> dict
:canonical: optim.scheduler.OptimizerParamScheduler.state_dict

```{autodoc2-docstring} optim.scheduler.OptimizerParamScheduler.state_dict
```

````

````{py:method} _check_and_set(cls_value: float, sd_value: float, name: str) -> float
:canonical: optim.scheduler.OptimizerParamScheduler._check_and_set

```{autodoc2-docstring} optim.scheduler.OptimizerParamScheduler._check_and_set
```

````

````{py:method} load_state_dict(state_dict: dict) -> None
:canonical: optim.scheduler.OptimizerParamScheduler.load_state_dict

```{autodoc2-docstring} optim.scheduler.OptimizerParamScheduler.load_state_dict
```

````

`````
