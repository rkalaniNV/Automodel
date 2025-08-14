# {py:mod}`nemo_automodel.components.datasets.llm.mock`

```{py:module} nemo_automodel.components.datasets.llm.mock
```

```{autodoc2-docstring} nemo_automodel.components.datasets.llm.mock
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`make_vocab <nemo_automodel.components.datasets.llm.mock.make_vocab>`
  - ```{autodoc2-docstring} nemo_automodel.components.datasets.llm.mock.make_vocab
    :summary:
    ```
* - {py:obj}`gen_sentence_ids <nemo_automodel.components.datasets.llm.mock.gen_sentence_ids>`
  - ```{autodoc2-docstring} nemo_automodel.components.datasets.llm.mock.gen_sentence_ids
    :summary:
    ```
* - {py:obj}`build_unpacked_dataset <nemo_automodel.components.datasets.llm.mock.build_unpacked_dataset>`
  - ```{autodoc2-docstring} nemo_automodel.components.datasets.llm.mock.build_unpacked_dataset
    :summary:
    ```
````

### API

````{py:function} make_vocab(vocab_size: int = 100)
:canonical: nemo_automodel.components.datasets.llm.mock.make_vocab

```{autodoc2-docstring} nemo_automodel.components.datasets.llm.mock.make_vocab
```
````

````{py:function} gen_sentence_ids(vocab, mean_len: float, std_len: float, max_len: int)
:canonical: nemo_automodel.components.datasets.llm.mock.gen_sentence_ids

```{autodoc2-docstring} nemo_automodel.components.datasets.llm.mock.gen_sentence_ids
```
````

````{py:function} build_unpacked_dataset(*, num_sentences: int = 10, mean_len: float = 20.0, std_len: float = 6.0, vocab_size: int = 100, max_sentence_len: int = 64, seed: int = 0, tokenizer=None)
:canonical: nemo_automodel.components.datasets.llm.mock.build_unpacked_dataset

```{autodoc2-docstring} nemo_automodel.components.datasets.llm.mock.build_unpacked_dataset
```
````
