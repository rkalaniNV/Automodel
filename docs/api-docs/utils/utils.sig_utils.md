# {py:mod}`utils.sig_utils`

```{py:module} utils.sig_utils
```

```{autodoc2-docstring} utils.sig_utils
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DistributedSignalHandler <utils.sig_utils.DistributedSignalHandler>`
  - ```{autodoc2-docstring} utils.sig_utils.DistributedSignalHandler
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_device <utils.sig_utils.get_device>`
  - ```{autodoc2-docstring} utils.sig_utils.get_device
    :summary:
    ```
* - {py:obj}`all_gather_item <utils.sig_utils.all_gather_item>`
  - ```{autodoc2-docstring} utils.sig_utils.all_gather_item
    :summary:
    ```
````

### API

````{py:function} get_device(local_rank: typing.Optional[int] = None) -> torch.device
:canonical: utils.sig_utils.get_device

```{autodoc2-docstring} utils.sig_utils.get_device
```
````

````{py:function} all_gather_item(item: typing.Any, dtype: torch.dtype, group: typing.Optional[torch.distributed.ProcessGroup] = None, async_op: bool = False, local_rank: typing.Optional[int] = None) -> list[typing.Any]
:canonical: utils.sig_utils.all_gather_item

```{autodoc2-docstring} utils.sig_utils.all_gather_item
```
````

`````{py:class} DistributedSignalHandler(sig: int = signal.SIGTERM)
:canonical: utils.sig_utils.DistributedSignalHandler

```{autodoc2-docstring} utils.sig_utils.DistributedSignalHandler
```

```{rubric} Initialization
```

```{autodoc2-docstring} utils.sig_utils.DistributedSignalHandler.__init__
```

````{py:method} signals_received() -> list[bool]
:canonical: utils.sig_utils.DistributedSignalHandler.signals_received

```{autodoc2-docstring} utils.sig_utils.DistributedSignalHandler.signals_received
```

````

````{py:method} __enter__() -> utils.sig_utils.DistributedSignalHandler
:canonical: utils.sig_utils.DistributedSignalHandler.__enter__

```{autodoc2-docstring} utils.sig_utils.DistributedSignalHandler.__enter__
```

````

````{py:method} __exit__(exc_type: typing.Optional[type], exc_val: BaseException | None, exc_tb: types.TracebackType | None) -> None
:canonical: utils.sig_utils.DistributedSignalHandler.__exit__

```{autodoc2-docstring} utils.sig_utils.DistributedSignalHandler.__exit__
```

````

````{py:method} release() -> bool
:canonical: utils.sig_utils.DistributedSignalHandler.release

```{autodoc2-docstring} utils.sig_utils.DistributedSignalHandler.release
```

````

`````
