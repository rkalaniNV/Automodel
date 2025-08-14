# {py:mod}`nemo_automodel.components.datasets.llm.packed_sequence`

```{py:module} nemo_automodel.components.datasets.llm.packed_sequence
```

```{autodoc2-docstring} nemo_automodel.components.datasets.llm.packed_sequence
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PackedSequence <nemo_automodel.components.datasets.llm.packed_sequence.PackedSequence>`
  - ```{autodoc2-docstring} nemo_automodel.components.datasets.llm.packed_sequence.PackedSequence
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_block_causal_mask <nemo_automodel.components.datasets.llm.packed_sequence.create_block_causal_mask>`
  - ```{autodoc2-docstring} nemo_automodel.components.datasets.llm.packed_sequence.create_block_causal_mask
    :summary:
    ```
* - {py:obj}`packed_block_causal_mask <nemo_automodel.components.datasets.llm.packed_sequence.packed_block_causal_mask>`
  - ```{autodoc2-docstring} nemo_automodel.components.datasets.llm.packed_sequence.packed_block_causal_mask
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <nemo_automodel.components.datasets.llm.packed_sequence.logger>`
  - ```{autodoc2-docstring} nemo_automodel.components.datasets.llm.packed_sequence.logger
    :summary:
    ```
* - {py:obj}`CROSS_ENTROPY_IGNORE_IDX <nemo_automodel.components.datasets.llm.packed_sequence.CROSS_ENTROPY_IGNORE_IDX>`
  - ```{autodoc2-docstring} nemo_automodel.components.datasets.llm.packed_sequence.CROSS_ENTROPY_IGNORE_IDX
    :summary:
    ```
* - {py:obj}`PACK_TYPE <nemo_automodel.components.datasets.llm.packed_sequence.PACK_TYPE>`
  - ```{autodoc2-docstring} nemo_automodel.components.datasets.llm.packed_sequence.PACK_TYPE
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: nemo_automodel.components.datasets.llm.packed_sequence.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} nemo_automodel.components.datasets.llm.packed_sequence.logger
```

````

````{py:data} CROSS_ENTROPY_IGNORE_IDX
:canonical: nemo_automodel.components.datasets.llm.packed_sequence.CROSS_ENTROPY_IGNORE_IDX
:value: >
   None

```{autodoc2-docstring} nemo_automodel.components.datasets.llm.packed_sequence.CROSS_ENTROPY_IGNORE_IDX
```

````

````{py:data} PACK_TYPE
:canonical: nemo_automodel.components.datasets.llm.packed_sequence.PACK_TYPE
:value: >
   None

```{autodoc2-docstring} nemo_automodel.components.datasets.llm.packed_sequence.PACK_TYPE
```

````

`````{py:class} PackedSequence(dataset, split, packed_sequence_size, split_across_pack=False, max_packs=None)
:canonical: nemo_automodel.components.datasets.llm.packed_sequence.PackedSequence

```{autodoc2-docstring} nemo_automodel.components.datasets.llm.packed_sequence.PackedSequence
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_automodel.components.datasets.llm.packed_sequence.PackedSequence.__init__
```

````{py:method} pack()
:canonical: nemo_automodel.components.datasets.llm.packed_sequence.PackedSequence.pack

```{autodoc2-docstring} nemo_automodel.components.datasets.llm.packed_sequence.PackedSequence.pack
```

````

````{py:method} _should_stop_packing() -> bool
:canonical: nemo_automodel.components.datasets.llm.packed_sequence.PackedSequence._should_stop_packing

```{autodoc2-docstring} nemo_automodel.components.datasets.llm.packed_sequence.PackedSequence._should_stop_packing
```

````

````{py:method} _split_and_add_pack(current_pack: nemo_automodel.components.datasets.llm.packed_sequence.PACK_TYPE) -> nemo_automodel.components.datasets.llm.packed_sequence.PACK_TYPE
:canonical: nemo_automodel.components.datasets.llm.packed_sequence.PackedSequence._split_and_add_pack

```{autodoc2-docstring} nemo_automodel.components.datasets.llm.packed_sequence.PackedSequence._split_and_add_pack
```

````

````{py:method} _add_pack(pack: nemo_automodel.components.datasets.llm.packed_sequence.PACK_TYPE) -> None
:canonical: nemo_automodel.components.datasets.llm.packed_sequence.PackedSequence._add_pack

```{autodoc2-docstring} nemo_automodel.components.datasets.llm.packed_sequence.PackedSequence._add_pack
```

````

````{py:method} _convert_to_tensors(pack: nemo_automodel.components.datasets.llm.packed_sequence.PACK_TYPE) -> nemo_automodel.components.datasets.llm.packed_sequence.PACK_TYPE
:canonical: nemo_automodel.components.datasets.llm.packed_sequence.PackedSequence._convert_to_tensors

```{autodoc2-docstring} nemo_automodel.components.datasets.llm.packed_sequence.PackedSequence._convert_to_tensors
```

````

````{py:method} _pad_pack(pack: nemo_automodel.components.datasets.llm.packed_sequence.PACK_TYPE, padding_idx: int) -> nemo_automodel.components.datasets.llm.packed_sequence.PACK_TYPE
:canonical: nemo_automodel.components.datasets.llm.packed_sequence.PackedSequence._pad_pack

```{autodoc2-docstring} nemo_automodel.components.datasets.llm.packed_sequence.PackedSequence._pad_pack
```

````

`````

````{py:function} create_block_causal_mask(seq_lens: list[torch.Tensor]) -> torch.Tensor
:canonical: nemo_automodel.components.datasets.llm.packed_sequence.create_block_causal_mask

```{autodoc2-docstring} nemo_automodel.components.datasets.llm.packed_sequence.create_block_causal_mask
```
````

````{py:function} packed_block_causal_mask(seq_lens: list[torch.Tensor])
:canonical: nemo_automodel.components.datasets.llm.packed_sequence.packed_block_causal_mask

```{autodoc2-docstring} nemo_automodel.components.datasets.llm.packed_sequence.packed_block_causal_mask
```
````
