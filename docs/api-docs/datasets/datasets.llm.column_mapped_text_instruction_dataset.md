# {py:mod}`datasets.llm.column_mapped_text_instruction_dataset`

```{py:module} datasets.llm.column_mapped_text_instruction_dataset
```

```{autodoc2-docstring} datasets.llm.column_mapped_text_instruction_dataset
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ColumnTypes <datasets.llm.column_mapped_text_instruction_dataset.ColumnTypes>`
  -
* - {py:obj}`ColumnMappedTextInstructionDataset <datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset>`
  - ```{autodoc2-docstring} datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`make_iterable <datasets.llm.column_mapped_text_instruction_dataset.make_iterable>`
  - ```{autodoc2-docstring} datasets.llm.column_mapped_text_instruction_dataset.make_iterable
    :summary:
    ```
* - {py:obj}`_str_is_hf_repo_id <datasets.llm.column_mapped_text_instruction_dataset._str_is_hf_repo_id>`
  - ```{autodoc2-docstring} datasets.llm.column_mapped_text_instruction_dataset._str_is_hf_repo_id
    :summary:
    ```
* - {py:obj}`_load_dataset <datasets.llm.column_mapped_text_instruction_dataset._load_dataset>`
  - ```{autodoc2-docstring} datasets.llm.column_mapped_text_instruction_dataset._load_dataset
    :summary:
    ```
* - {py:obj}`_apply_tokenizer_with_chat_template <datasets.llm.column_mapped_text_instruction_dataset._apply_tokenizer_with_chat_template>`
  - ```{autodoc2-docstring} datasets.llm.column_mapped_text_instruction_dataset._apply_tokenizer_with_chat_template
    :summary:
    ```
* - {py:obj}`_apply_tokenizer_plain <datasets.llm.column_mapped_text_instruction_dataset._apply_tokenizer_plain>`
  - ```{autodoc2-docstring} datasets.llm.column_mapped_text_instruction_dataset._apply_tokenizer_plain
    :summary:
    ```
* - {py:obj}`_has_chat_template <datasets.llm.column_mapped_text_instruction_dataset._has_chat_template>`
  - ```{autodoc2-docstring} datasets.llm.column_mapped_text_instruction_dataset._has_chat_template
    :summary:
    ```
* - {py:obj}`_check_all_values_equal_length <datasets.llm.column_mapped_text_instruction_dataset._check_all_values_equal_length>`
  - ```{autodoc2-docstring} datasets.llm.column_mapped_text_instruction_dataset._check_all_values_equal_length
    :summary:
    ```
````

### API

`````{py:class} ColumnTypes(*args, **kwds)
:canonical: datasets.llm.column_mapped_text_instruction_dataset.ColumnTypes

Bases: {py:obj}`enum.Enum`

````{py:attribute} Context
:canonical: datasets.llm.column_mapped_text_instruction_dataset.ColumnTypes.Context
:value: >
   'context'

```{autodoc2-docstring} datasets.llm.column_mapped_text_instruction_dataset.ColumnTypes.Context
```

````

````{py:attribute} Question
:canonical: datasets.llm.column_mapped_text_instruction_dataset.ColumnTypes.Question
:value: >
   'question'

```{autodoc2-docstring} datasets.llm.column_mapped_text_instruction_dataset.ColumnTypes.Question
```

````

````{py:attribute} Answer
:canonical: datasets.llm.column_mapped_text_instruction_dataset.ColumnTypes.Answer
:value: >
   'answer'

```{autodoc2-docstring} datasets.llm.column_mapped_text_instruction_dataset.ColumnTypes.Answer
```

````

`````

````{py:function} make_iterable(val: typing.Union[str, typing.List[str]]) -> typing.Iterator[str]
:canonical: datasets.llm.column_mapped_text_instruction_dataset.make_iterable

```{autodoc2-docstring} datasets.llm.column_mapped_text_instruction_dataset.make_iterable
```
````

````{py:function} _str_is_hf_repo_id(val: str) -> bool
:canonical: datasets.llm.column_mapped_text_instruction_dataset._str_is_hf_repo_id

```{autodoc2-docstring} datasets.llm.column_mapped_text_instruction_dataset._str_is_hf_repo_id
```
````

````{py:function} _load_dataset(path_or_dataset_id: typing.Union[str, typing.List[str]], split: typing.Optional[str] = None, streaming: bool = False)
:canonical: datasets.llm.column_mapped_text_instruction_dataset._load_dataset

```{autodoc2-docstring} datasets.llm.column_mapped_text_instruction_dataset._load_dataset
```
````

````{py:function} _apply_tokenizer_with_chat_template(tokenizer: transformers.PreTrainedTokenizer, context: str, question: str, answer: str, start_of_turn_token: typing.Optional[str] = None, answer_only_loss_mask: bool = True) -> typing.Dict[str, typing.List[int]]
:canonical: datasets.llm.column_mapped_text_instruction_dataset._apply_tokenizer_with_chat_template

```{autodoc2-docstring} datasets.llm.column_mapped_text_instruction_dataset._apply_tokenizer_with_chat_template
```
````

````{py:function} _apply_tokenizer_plain(tokenizer: transformers.PreTrainedTokenizer, context: str, question: str, answer: str, answer_only_loss_mask: bool = True) -> typing.Dict[str, typing.List[int]]
:canonical: datasets.llm.column_mapped_text_instruction_dataset._apply_tokenizer_plain

```{autodoc2-docstring} datasets.llm.column_mapped_text_instruction_dataset._apply_tokenizer_plain
```
````

````{py:function} _has_chat_template(tokenizer: transformers.PreTrainedTokenizer) -> bool
:canonical: datasets.llm.column_mapped_text_instruction_dataset._has_chat_template

```{autodoc2-docstring} datasets.llm.column_mapped_text_instruction_dataset._has_chat_template
```
````

````{py:function} _check_all_values_equal_length(sample: typing.Dict[str, typing.List[int]]) -> bool
:canonical: datasets.llm.column_mapped_text_instruction_dataset._check_all_values_equal_length

```{autodoc2-docstring} datasets.llm.column_mapped_text_instruction_dataset._check_all_values_equal_length
```
````

`````{py:class} ColumnMappedTextInstructionDataset(path_or_dataset_id: typing.Union[str, typing.List[str]], column_mapping: typing.Dict[str, str], tokenizer, *, split: typing.Optional[str] = None, streaming: bool = False, answer_only_loss_mask: bool = True, start_of_turn_token: typing.Optional[str] = None)
:canonical: datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset

Bases: {py:obj}`torch.utils.data.Dataset`

```{autodoc2-docstring} datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
```

```{rubric} Initialization
```

```{autodoc2-docstring} datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset.__init__
```

````{py:method} __len__() -> int
:canonical: datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset.__len__

```{autodoc2-docstring} datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset.__len__
```

````

````{py:method} __getitem__(idx)
:canonical: datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset.__getitem__

```{autodoc2-docstring} datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset.__getitem__
```

````

````{py:method} __iter__()
:canonical: datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset.__iter__

```{autodoc2-docstring} datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset.__iter__
```

````

````{py:method} _apply_tokenizer(sample: typing.Dict[str, str]) -> typing.Dict[str, typing.List[int]]
:canonical: datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset._apply_tokenizer

```{autodoc2-docstring} datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset._apply_tokenizer
```

````

`````
