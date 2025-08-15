# {py:mod}`datasets.utils`

```{py:module} datasets.utils
```

```{autodoc2-docstring} datasets.utils
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SFTSingleTurnPreprocessor <datasets.utils.SFTSingleTurnPreprocessor>`
  - ```{autodoc2-docstring} datasets.utils.SFTSingleTurnPreprocessor
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`batchify <datasets.utils.batchify>`
  - ```{autodoc2-docstring} datasets.utils.batchify
    :summary:
    ```
* - {py:obj}`extract_key_from_dicts <datasets.utils.extract_key_from_dicts>`
  - ```{autodoc2-docstring} datasets.utils.extract_key_from_dicts
    :summary:
    ```
* - {py:obj}`pad_within_micro <datasets.utils.pad_within_micro>`
  - ```{autodoc2-docstring} datasets.utils.pad_within_micro
    :summary:
    ```
* - {py:obj}`default_collater <datasets.utils.default_collater>`
  - ```{autodoc2-docstring} datasets.utils.default_collater
    :summary:
    ```
````

### API

````{py:function} batchify(tensor)
:canonical: datasets.utils.batchify

```{autodoc2-docstring} datasets.utils.batchify
```
````

````{py:function} extract_key_from_dicts(batch, key)
:canonical: datasets.utils.extract_key_from_dicts

```{autodoc2-docstring} datasets.utils.extract_key_from_dicts
```
````

````{py:function} pad_within_micro(batch, pad_token_id, pad_seq_len_divisible=None)
:canonical: datasets.utils.pad_within_micro

```{autodoc2-docstring} datasets.utils.pad_within_micro
```
````

````{py:function} default_collater(batch, pad_token_id=0, pad_seq_len_divisible=None)
:canonical: datasets.utils.default_collater

```{autodoc2-docstring} datasets.utils.default_collater
```
````

`````{py:class} SFTSingleTurnPreprocessor(tokenizer)
:canonical: datasets.utils.SFTSingleTurnPreprocessor

```{autodoc2-docstring} datasets.utils.SFTSingleTurnPreprocessor
```

```{rubric} Initialization
```

```{autodoc2-docstring} datasets.utils.SFTSingleTurnPreprocessor.__init__
```

````{py:method} _tokenize_function(examples, dataset)
:canonical: datasets.utils.SFTSingleTurnPreprocessor._tokenize_function

```{autodoc2-docstring} datasets.utils.SFTSingleTurnPreprocessor._tokenize_function
```

````

````{py:method} _compute_dataset_max_len(tokenized_ds)
:canonical: datasets.utils.SFTSingleTurnPreprocessor._compute_dataset_max_len

```{autodoc2-docstring} datasets.utils.SFTSingleTurnPreprocessor._compute_dataset_max_len
```

````

````{py:method} _pad_function(max_len)
:canonical: datasets.utils.SFTSingleTurnPreprocessor._pad_function

```{autodoc2-docstring} datasets.utils.SFTSingleTurnPreprocessor._pad_function
```

````

````{py:method} process(raw_dataset, ds)
:canonical: datasets.utils.SFTSingleTurnPreprocessor.process

```{autodoc2-docstring} datasets.utils.SFTSingleTurnPreprocessor.process
```

````

`````
