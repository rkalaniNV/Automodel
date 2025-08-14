# {py:mod}`nemo_automodel.components.loggers.log_utils`

```{py:module} nemo_automodel.components.loggers.log_utils
```

```{autodoc2-docstring} nemo_automodel.components.loggers.log_utils
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RankFilter <nemo_automodel.components.loggers.log_utils.RankFilter>`
  - ```{autodoc2-docstring} nemo_automodel.components.loggers.log_utils.RankFilter
    :summary:
    ```
* - {py:obj}`ColorFormatter <nemo_automodel.components.loggers.log_utils.ColorFormatter>`
  - ```{autodoc2-docstring} nemo_automodel.components.loggers.log_utils.ColorFormatter
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`warning_filter <nemo_automodel.components.loggers.log_utils.warning_filter>`
  - ```{autodoc2-docstring} nemo_automodel.components.loggers.log_utils.warning_filter
    :summary:
    ```
* - {py:obj}`module_filter <nemo_automodel.components.loggers.log_utils.module_filter>`
  - ```{autodoc2-docstring} nemo_automodel.components.loggers.log_utils.module_filter
    :summary:
    ```
* - {py:obj}`add_filter_to_all_loggers <nemo_automodel.components.loggers.log_utils.add_filter_to_all_loggers>`
  - ```{autodoc2-docstring} nemo_automodel.components.loggers.log_utils.add_filter_to_all_loggers
    :summary:
    ```
* - {py:obj}`_ensure_root_handler_with_formatter <nemo_automodel.components.loggers.log_utils._ensure_root_handler_with_formatter>`
  - ```{autodoc2-docstring} nemo_automodel.components.loggers.log_utils._ensure_root_handler_with_formatter
    :summary:
    ```
* - {py:obj}`setup_logging <nemo_automodel.components.loggers.log_utils.setup_logging>`
  - ```{autodoc2-docstring} nemo_automodel.components.loggers.log_utils.setup_logging
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <nemo_automodel.components.loggers.log_utils.logger>`
  - ```{autodoc2-docstring} nemo_automodel.components.loggers.log_utils.logger
    :summary:
    ```
* - {py:obj}`_COLOR_RESET <nemo_automodel.components.loggers.log_utils._COLOR_RESET>`
  - ```{autodoc2-docstring} nemo_automodel.components.loggers.log_utils._COLOR_RESET
    :summary:
    ```
* - {py:obj}`_LEVEL_TO_COLOR <nemo_automodel.components.loggers.log_utils._LEVEL_TO_COLOR>`
  - ```{autodoc2-docstring} nemo_automodel.components.loggers.log_utils._LEVEL_TO_COLOR
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: nemo_automodel.components.loggers.log_utils.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} nemo_automodel.components.loggers.log_utils.logger
```

````

`````{py:class} RankFilter(name='')
:canonical: nemo_automodel.components.loggers.log_utils.RankFilter

Bases: {py:obj}`logging.Filter`

```{autodoc2-docstring} nemo_automodel.components.loggers.log_utils.RankFilter
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_automodel.components.loggers.log_utils.RankFilter.__init__
```

````{py:method} filter(record)
:canonical: nemo_automodel.components.loggers.log_utils.RankFilter.filter

```{autodoc2-docstring} nemo_automodel.components.loggers.log_utils.RankFilter.filter
```

````

`````

````{py:data} _COLOR_RESET
:canonical: nemo_automodel.components.loggers.log_utils._COLOR_RESET
:value: >
   '\x1b[0m'

```{autodoc2-docstring} nemo_automodel.components.loggers.log_utils._COLOR_RESET
```

````

````{py:data} _LEVEL_TO_COLOR
:canonical: nemo_automodel.components.loggers.log_utils._LEVEL_TO_COLOR
:value: >
   None

```{autodoc2-docstring} nemo_automodel.components.loggers.log_utils._LEVEL_TO_COLOR
```

````

`````{py:class} ColorFormatter(fmt: str | None = None, datefmt: str | None = None, use_color: bool = True)
:canonical: nemo_automodel.components.loggers.log_utils.ColorFormatter

Bases: {py:obj}`logging.Formatter`

```{autodoc2-docstring} nemo_automodel.components.loggers.log_utils.ColorFormatter
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_automodel.components.loggers.log_utils.ColorFormatter.__init__
```

````{py:method} _stream_supports_color() -> bool
:canonical: nemo_automodel.components.loggers.log_utils.ColorFormatter._stream_supports_color

```{autodoc2-docstring} nemo_automodel.components.loggers.log_utils.ColorFormatter._stream_supports_color
```

````

````{py:method} format(record: logging.LogRecord) -> str
:canonical: nemo_automodel.components.loggers.log_utils.ColorFormatter.format

````

`````

````{py:function} warning_filter(record: logging.LogRecord) -> bool
:canonical: nemo_automodel.components.loggers.log_utils.warning_filter

```{autodoc2-docstring} nemo_automodel.components.loggers.log_utils.warning_filter
```
````

````{py:function} module_filter(record: logging.LogRecord, modules_to_filter: list[str]) -> bool
:canonical: nemo_automodel.components.loggers.log_utils.module_filter

```{autodoc2-docstring} nemo_automodel.components.loggers.log_utils.module_filter
```
````

````{py:function} add_filter_to_all_loggers(filter: typing.Union[logging.Filter, typing.Callable[[logging.LogRecord], bool]]) -> None
:canonical: nemo_automodel.components.loggers.log_utils.add_filter_to_all_loggers

```{autodoc2-docstring} nemo_automodel.components.loggers.log_utils.add_filter_to_all_loggers
```
````

````{py:function} _ensure_root_handler_with_formatter(formatter: logging.Formatter) -> None
:canonical: nemo_automodel.components.loggers.log_utils._ensure_root_handler_with_formatter

```{autodoc2-docstring} nemo_automodel.components.loggers.log_utils._ensure_root_handler_with_formatter
```
````

````{py:function} setup_logging(logging_level: int = logging.INFO, filter_warning: bool = True, modules_to_filter: typing.Optional[list[str]] = None, set_level_for_all_loggers: bool = False) -> None
:canonical: nemo_automodel.components.loggers.log_utils.setup_logging

```{autodoc2-docstring} nemo_automodel.components.loggers.log_utils.setup_logging
```
````
