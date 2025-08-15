# {py:mod}`datasets.llm.mock_packed`

```{py:module} datasets.llm.mock_packed
```

```{autodoc2-docstring} datasets.llm.mock_packed
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`make_vocab <datasets.llm.mock_packed.make_vocab>`
  - ```{autodoc2-docstring} datasets.llm.mock_packed.make_vocab
    :summary:
    ```
* - {py:obj}`gen_sentence_ids <datasets.llm.mock_packed.gen_sentence_ids>`
  - ```{autodoc2-docstring} datasets.llm.mock_packed.gen_sentence_ids
    :summary:
    ```
* - {py:obj}`flush_block <datasets.llm.mock_packed.flush_block>`
  - ```{autodoc2-docstring} datasets.llm.mock_packed.flush_block
    :summary:
    ```
* - {py:obj}`build_packed_dataset <datasets.llm.mock_packed.build_packed_dataset>`
  - ```{autodoc2-docstring} datasets.llm.mock_packed.build_packed_dataset
    :summary:
    ```
````

### API

````{py:function} make_vocab(vocab_size: int = 100)
:canonical: datasets.llm.mock_packed.make_vocab

```{autodoc2-docstring} datasets.llm.mock_packed.make_vocab
```
````

````{py:function} gen_sentence_ids(vocab, mean_len: float, std_len: float, max_len: int)
:canonical: datasets.llm.mock_packed.gen_sentence_ids

```{autodoc2-docstring} datasets.llm.mock_packed.gen_sentence_ids
```
````

````{py:function} flush_block(block, block_size)
:canonical: datasets.llm.mock_packed.flush_block

```{autodoc2-docstring} datasets.llm.mock_packed.flush_block
```
````

````{py:function} build_packed_dataset(*, num_blocks: int = 10, block_size: int = 128, mean_len: float = 20.0, std_len: float = 6.0, vocab_size: int = 100, max_sentence_len: int = 64, seed: int = 0, tokenizer=None)
:canonical: datasets.llm.mock_packed.build_packed_dataset

```{autodoc2-docstring} datasets.llm.mock_packed.build_packed_dataset
```
````
