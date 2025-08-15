# {py:mod}`datasets.llm.mock`

```{py:module} datasets.llm.mock
```

```{autodoc2-docstring} datasets.llm.mock
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`make_vocab <datasets.llm.mock.make_vocab>`
  - ```{autodoc2-docstring} datasets.llm.mock.make_vocab
    :summary:
    ```
* - {py:obj}`gen_sentence_ids <datasets.llm.mock.gen_sentence_ids>`
  - ```{autodoc2-docstring} datasets.llm.mock.gen_sentence_ids
    :summary:
    ```
* - {py:obj}`build_unpacked_dataset <datasets.llm.mock.build_unpacked_dataset>`
  - ```{autodoc2-docstring} datasets.llm.mock.build_unpacked_dataset
    :summary:
    ```
````

### API

````{py:function} make_vocab(vocab_size: int = 100)
:canonical: datasets.llm.mock.make_vocab

```{autodoc2-docstring} datasets.llm.mock.make_vocab
```
````

````{py:function} gen_sentence_ids(vocab, mean_len: float, std_len: float, max_len: int)
:canonical: datasets.llm.mock.gen_sentence_ids

```{autodoc2-docstring} datasets.llm.mock.gen_sentence_ids
```
````

````{py:function} build_unpacked_dataset(*, num_sentences: int = 10, mean_len: float = 20.0, std_len: float = 6.0, vocab_size: int = 100, max_sentence_len: int = 64, seed: int = 0, tokenizer=None)
:canonical: datasets.llm.mock.build_unpacked_dataset

```{autodoc2-docstring} datasets.llm.mock.build_unpacked_dataset
```
````
