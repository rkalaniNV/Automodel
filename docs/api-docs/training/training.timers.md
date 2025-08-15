# {py:mod}`training.timers`

```{py:module} training.timers
```

```{autodoc2-docstring} training.timers
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TimerBase <training.timers.TimerBase>`
  - ```{autodoc2-docstring} training.timers.TimerBase
    :summary:
    ```
* - {py:obj}`DummyTimer <training.timers.DummyTimer>`
  - ```{autodoc2-docstring} training.timers.DummyTimer
    :summary:
    ```
* - {py:obj}`Timer <training.timers.Timer>`
  - ```{autodoc2-docstring} training.timers.Timer
    :summary:
    ```
* - {py:obj}`Timers <training.timers.Timers>`
  - ```{autodoc2-docstring} training.timers.Timers
    :summary:
    ```
````

### API

`````{py:class} TimerBase(name: str)
:canonical: training.timers.TimerBase

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} training.timers.TimerBase
```

```{rubric} Initialization
```

```{autodoc2-docstring} training.timers.TimerBase.__init__
```

````{py:method} with_barrier(barrier=True)
:canonical: training.timers.TimerBase.with_barrier

```{autodoc2-docstring} training.timers.TimerBase.with_barrier
```

````

````{py:method} __enter__()
:canonical: training.timers.TimerBase.__enter__

```{autodoc2-docstring} training.timers.TimerBase.__enter__
```

````

````{py:method} __exit__(exc_type, exc_val, exc_tb)
:canonical: training.timers.TimerBase.__exit__

```{autodoc2-docstring} training.timers.TimerBase.__exit__
```

````

````{py:method} start(barrier=False)
:canonical: training.timers.TimerBase.start
:abstractmethod:

```{autodoc2-docstring} training.timers.TimerBase.start
```

````

````{py:method} stop(barrier=False)
:canonical: training.timers.TimerBase.stop
:abstractmethod:

```{autodoc2-docstring} training.timers.TimerBase.stop
```

````

````{py:method} reset()
:canonical: training.timers.TimerBase.reset
:abstractmethod:

```{autodoc2-docstring} training.timers.TimerBase.reset
```

````

````{py:method} elapsed(reset=True, barrier=False)
:canonical: training.timers.TimerBase.elapsed
:abstractmethod:

```{autodoc2-docstring} training.timers.TimerBase.elapsed
```

````

`````

`````{py:class} DummyTimer()
:canonical: training.timers.DummyTimer

Bases: {py:obj}`training.timers.TimerBase`

```{autodoc2-docstring} training.timers.DummyTimer
```

```{rubric} Initialization
```

```{autodoc2-docstring} training.timers.DummyTimer.__init__
```

````{py:method} start(barrier=False)
:canonical: training.timers.DummyTimer.start

```{autodoc2-docstring} training.timers.DummyTimer.start
```

````

````{py:method} stop(barrier=False)
:canonical: training.timers.DummyTimer.stop

```{autodoc2-docstring} training.timers.DummyTimer.stop
```

````

````{py:method} reset()
:canonical: training.timers.DummyTimer.reset

```{autodoc2-docstring} training.timers.DummyTimer.reset
```

````

````{py:method} elapsed(reset=True, barrier=False)
:canonical: training.timers.DummyTimer.elapsed

```{autodoc2-docstring} training.timers.DummyTimer.elapsed
```

````

````{py:method} active_time()
:canonical: training.timers.DummyTimer.active_time

```{autodoc2-docstring} training.timers.DummyTimer.active_time
```

````

`````

`````{py:class} Timer(name)
:canonical: training.timers.Timer

Bases: {py:obj}`training.timers.TimerBase`

```{autodoc2-docstring} training.timers.Timer
```

```{rubric} Initialization
```

```{autodoc2-docstring} training.timers.Timer.__init__
```

````{py:method} set_barrier_group(barrier_group)
:canonical: training.timers.Timer.set_barrier_group

```{autodoc2-docstring} training.timers.Timer.set_barrier_group
```

````

````{py:method} start(barrier=False)
:canonical: training.timers.Timer.start

```{autodoc2-docstring} training.timers.Timer.start
```

````

````{py:method} stop(barrier=False)
:canonical: training.timers.Timer.stop

```{autodoc2-docstring} training.timers.Timer.stop
```

````

````{py:method} reset()
:canonical: training.timers.Timer.reset

```{autodoc2-docstring} training.timers.Timer.reset
```

````

````{py:method} elapsed(reset=True, barrier=False)
:canonical: training.timers.Timer.elapsed

```{autodoc2-docstring} training.timers.Timer.elapsed
```

````

````{py:method} active_time()
:canonical: training.timers.Timer.active_time

```{autodoc2-docstring} training.timers.Timer.active_time
```

````

`````

`````{py:class} Timers(log_level, log_option)
:canonical: training.timers.Timers

```{autodoc2-docstring} training.timers.Timers
```

```{rubric} Initialization
```

```{autodoc2-docstring} training.timers.Timers.__init__
```

````{py:method} __call__(name, log_level=None, barrier=False)
:canonical: training.timers.Timers.__call__

```{autodoc2-docstring} training.timers.Timers.__call__
```

````

````{py:method} _get_elapsed_time_all_ranks(names, reset, barrier)
:canonical: training.timers.Timers._get_elapsed_time_all_ranks

```{autodoc2-docstring} training.timers.Timers._get_elapsed_time_all_ranks
```

````

````{py:method} _get_global_min_max_time(names, reset, barrier, normalizer)
:canonical: training.timers.Timers._get_global_min_max_time

```{autodoc2-docstring} training.timers.Timers._get_global_min_max_time
```

````

````{py:method} _get_global_min_max_time_string(names, reset, barrier, normalizer, max_only)
:canonical: training.timers.Timers._get_global_min_max_time_string

```{autodoc2-docstring} training.timers.Timers._get_global_min_max_time_string
```

````

````{py:method} _get_all_ranks_time_string(names, reset, barrier, normalizer)
:canonical: training.timers.Timers._get_all_ranks_time_string

```{autodoc2-docstring} training.timers.Timers._get_all_ranks_time_string
```

````

````{py:method} get_all_timers_string(names: typing.List[str] = None, normalizer: float = 1.0, reset: bool = True, barrier: bool = False)
:canonical: training.timers.Timers.get_all_timers_string

```{autodoc2-docstring} training.timers.Timers.get_all_timers_string
```

````

````{py:method} log(names: typing.List[str], rank: int = None, normalizer: float = 1.0, reset: bool = True, barrier: bool = False)
:canonical: training.timers.Timers.log

```{autodoc2-docstring} training.timers.Timers.log
```

````

````{py:method} write(names: typing.List[str], writer, iteration: int, normalizer: float = 1.0, reset: bool = True, barrier: bool = False)
:canonical: training.timers.Timers.write

```{autodoc2-docstring} training.timers.Timers.write
```

````

````{py:method} write_to_wandb(names: list[str], writer, iteration: int, normalizer: float = 1.0, reset: bool = True, barrier: bool = False) -> None
:canonical: training.timers.Timers.write_to_wandb

```{autodoc2-docstring} training.timers.Timers.write_to_wandb
```

````

`````
