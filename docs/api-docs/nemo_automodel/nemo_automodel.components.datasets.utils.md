# {py:mod}`nemo_automodel.components.datasets.utils`

```{py:module} nemo_automodel.components.datasets.utils
```

```{autodoc2-docstring} nemo_automodel.components.datasets.utils
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SFTSingleTurnPreprocessor <nemo_automodel.components.datasets.utils.SFTSingleTurnPreprocessor>`
  - ```{autodoc2-docstring} nemo_automodel.components.datasets.utils.SFTSingleTurnPreprocessor
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`batchify <nemo_automodel.components.datasets.utils.batchify>`
  - ```{autodoc2-docstring} nemo_automodel.components.datasets.utils.batchify
    :summary:
    ```
* - {py:obj}`extract_key_from_dicts <nemo_automodel.components.datasets.utils.extract_key_from_dicts>`
  - ```{autodoc2-docstring} nemo_automodel.components.datasets.utils.extract_key_from_dicts
    :summary:
    ```
* - {py:obj}`pad_within_micro <nemo_automodel.components.datasets.utils.pad_within_micro>`
  - ```{autodoc2-docstring} nemo_automodel.components.datasets.utils.pad_within_micro
    :summary:
    ```
* - {py:obj}`default_collater <nemo_automodel.components.datasets.utils.default_collater>`
  - ```{autodoc2-docstring} nemo_automodel.components.datasets.utils.default_collater
    :summary:
    ```
````

### API

````{py:function} batchify(tensor)
:canonical: nemo_automodel.components.datasets.utils.batchify

```{autodoc2-docstring} nemo_automodel.components.datasets.utils.batchify
```
````

````{py:function} extract_key_from_dicts(batch, key)
:canonical: nemo_automodel.components.datasets.utils.extract_key_from_dicts

```{autodoc2-docstring} nemo_automodel.components.datasets.utils.extract_key_from_dicts
```
````

````{py:function} pad_within_micro(batch, pad_token_id, pad_seq_len_divisible=None)
:canonical: nemo_automodel.components.datasets.utils.pad_within_micro

```{autodoc2-docstring} nemo_automodel.components.datasets.utils.pad_within_micro
```
````

````{py:function} default_collater(batch, pad_token_id=0, pad_seq_len_divisible=None)
:canonical: nemo_automodel.components.datasets.utils.default_collater

```{autodoc2-docstring} nemo_automodel.components.datasets.utils.default_collater
```
````

`````{py:class} SFTSingleTurnPreprocessor(tokenizer)
:canonical: nemo_automodel.components.datasets.utils.SFTSingleTurnPreprocessor

```{autodoc2-docstring} nemo_automodel.components.datasets.utils.SFTSingleTurnPreprocessor
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_automodel.components.datasets.utils.SFTSingleTurnPreprocessor.__init__
```

````{py:method} _tokenize_function(examples, dataset)
:canonical: nemo_automodel.components.datasets.utils.SFTSingleTurnPreprocessor._tokenize_function

```{autodoc2-docstring} nemo_automodel.components.datasets.utils.SFTSingleTurnPreprocessor._tokenize_function
```

````

````{py:method} _compute_dataset_max_len(tokenized_ds)
:canonical: nemo_automodel.components.datasets.utils.SFTSingleTurnPreprocessor._compute_dataset_max_len

```{autodoc2-docstring} nemo_automodel.components.datasets.utils.SFTSingleTurnPreprocessor._compute_dataset_max_len
```

````

````{py:method} _pad_function(max_len)
:canonical: nemo_automodel.components.datasets.utils.SFTSingleTurnPreprocessor._pad_function

```{autodoc2-docstring} nemo_automodel.components.datasets.utils.SFTSingleTurnPreprocessor._pad_function
```

````

````{py:method} process(raw_dataset, ds)
:canonical: nemo_automodel.components.datasets.utils.SFTSingleTurnPreprocessor.process

```{autodoc2-docstring} nemo_automodel.components.datasets.utils.SFTSingleTurnPreprocessor.process
```

````

`````
