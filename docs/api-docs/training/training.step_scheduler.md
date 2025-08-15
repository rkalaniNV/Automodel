# {py:mod}`training.step_scheduler`

```{py:module} training.step_scheduler
```

```{autodoc2-docstring} training.step_scheduler
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`StepScheduler <training.step_scheduler.StepScheduler>`
  - ```{autodoc2-docstring} training.step_scheduler.StepScheduler
    :summary:
    ```
````

### API

`````{py:class} StepScheduler(grad_acc_steps: int, ckpt_every_steps: int, dataloader: typing.Optional[int], val_every_steps: typing.Optional[int] = None, start_step: int = 0, start_epoch: int = 0, num_epochs: int = 10, max_steps: typing.Optional[int] = None)
:canonical: training.step_scheduler.StepScheduler

Bases: {py:obj}`torch.distributed.checkpoint.stateful.Stateful`

```{autodoc2-docstring} training.step_scheduler.StepScheduler
```

```{rubric} Initialization
```

```{autodoc2-docstring} training.step_scheduler.StepScheduler.__init__
```

````{py:method} __iter__()
:canonical: training.step_scheduler.StepScheduler.__iter__

```{autodoc2-docstring} training.step_scheduler.StepScheduler.__iter__
```

````

````{py:method} set_epoch(epoch: int)
:canonical: training.step_scheduler.StepScheduler.set_epoch

```{autodoc2-docstring} training.step_scheduler.StepScheduler.set_epoch
```

````

````{py:property} is_optim_step
:canonical: training.step_scheduler.StepScheduler.is_optim_step

```{autodoc2-docstring} training.step_scheduler.StepScheduler.is_optim_step
```

````

````{py:property} is_val_step
:canonical: training.step_scheduler.StepScheduler.is_val_step

```{autodoc2-docstring} training.step_scheduler.StepScheduler.is_val_step
```

````

````{py:property} is_ckpt_step
:canonical: training.step_scheduler.StepScheduler.is_ckpt_step

```{autodoc2-docstring} training.step_scheduler.StepScheduler.is_ckpt_step
```

````

````{py:property} epochs
:canonical: training.step_scheduler.StepScheduler.epochs

```{autodoc2-docstring} training.step_scheduler.StepScheduler.epochs
```

````

````{py:method} state_dict()
:canonical: training.step_scheduler.StepScheduler.state_dict

```{autodoc2-docstring} training.step_scheduler.StepScheduler.state_dict
```

````

````{py:method} load_state_dict(s)
:canonical: training.step_scheduler.StepScheduler.load_state_dict

```{autodoc2-docstring} training.step_scheduler.StepScheduler.load_state_dict
```

````

`````
