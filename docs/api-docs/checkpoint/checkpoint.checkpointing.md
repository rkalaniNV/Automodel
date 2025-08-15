# {py:mod}`checkpoint.checkpointing`

```{py:module} checkpoint.checkpointing
```

```{autodoc2-docstring} checkpoint.checkpointing
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CheckpointingConfig <checkpoint.checkpointing.CheckpointingConfig>`
  - ```{autodoc2-docstring} checkpoint.checkpointing.CheckpointingConfig
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`save_model <checkpoint.checkpointing.save_model>`
  - ```{autodoc2-docstring} checkpoint.checkpointing.save_model
    :summary:
    ```
* - {py:obj}`load_model <checkpoint.checkpointing.load_model>`
  - ```{autodoc2-docstring} checkpoint.checkpointing.load_model
    :summary:
    ```
* - {py:obj}`save_optimizer <checkpoint.checkpointing.save_optimizer>`
  - ```{autodoc2-docstring} checkpoint.checkpointing.save_optimizer
    :summary:
    ```
* - {py:obj}`load_optimizer <checkpoint.checkpointing.load_optimizer>`
  - ```{autodoc2-docstring} checkpoint.checkpointing.load_optimizer
    :summary:
    ```
* - {py:obj}`_get_safetensors_index_path <checkpoint.checkpointing._get_safetensors_index_path>`
  - ```{autodoc2-docstring} checkpoint.checkpointing._get_safetensors_index_path
    :summary:
    ```
* - {py:obj}`_save_peft_adapters <checkpoint.checkpointing._save_peft_adapters>`
  - ```{autodoc2-docstring} checkpoint.checkpointing._save_peft_adapters
    :summary:
    ```
* - {py:obj}`_get_hf_peft_config <checkpoint.checkpointing._get_hf_peft_config>`
  - ```{autodoc2-docstring} checkpoint.checkpointing._get_hf_peft_config
    :summary:
    ```
* - {py:obj}`_get_automodel_peft_metadata <checkpoint.checkpointing._get_automodel_peft_metadata>`
  - ```{autodoc2-docstring} checkpoint.checkpointing._get_automodel_peft_metadata
    :summary:
    ```
* - {py:obj}`_extract_target_modules <checkpoint.checkpointing._extract_target_modules>`
  - ```{autodoc2-docstring} checkpoint.checkpointing._extract_target_modules
    :summary:
    ```
````

### API

`````{py:class} CheckpointingConfig
:canonical: checkpoint.checkpointing.CheckpointingConfig

```{autodoc2-docstring} checkpoint.checkpointing.CheckpointingConfig
```

````{py:attribute} enabled
:canonical: checkpoint.checkpointing.CheckpointingConfig.enabled
:type: bool
:value: >
   None

```{autodoc2-docstring} checkpoint.checkpointing.CheckpointingConfig.enabled
```

````

````{py:attribute} checkpoint_dir
:canonical: checkpoint.checkpointing.CheckpointingConfig.checkpoint_dir
:type: str | pathlib.Path
:value: >
   None

```{autodoc2-docstring} checkpoint.checkpointing.CheckpointingConfig.checkpoint_dir
```

````

````{py:attribute} model_save_format
:canonical: checkpoint.checkpointing.CheckpointingConfig.model_save_format
:type: nemo_automodel.components.checkpoint._backports.filesystem.SerializationFormat | str
:value: >
   None

```{autodoc2-docstring} checkpoint.checkpointing.CheckpointingConfig.model_save_format
```

````

````{py:attribute} model_cache_dir
:canonical: checkpoint.checkpointing.CheckpointingConfig.model_cache_dir
:type: str | pathlib.Path
:value: >
   None

```{autodoc2-docstring} checkpoint.checkpointing.CheckpointingConfig.model_cache_dir
```

````

````{py:attribute} model_repo_id
:canonical: checkpoint.checkpointing.CheckpointingConfig.model_repo_id
:type: str
:value: >
   None

```{autodoc2-docstring} checkpoint.checkpointing.CheckpointingConfig.model_repo_id
```

````

````{py:attribute} save_consolidated
:canonical: checkpoint.checkpointing.CheckpointingConfig.save_consolidated
:type: bool
:value: >
   None

```{autodoc2-docstring} checkpoint.checkpointing.CheckpointingConfig.save_consolidated
```

````

````{py:attribute} is_peft
:canonical: checkpoint.checkpointing.CheckpointingConfig.is_peft
:type: bool
:value: >
   None

```{autodoc2-docstring} checkpoint.checkpointing.CheckpointingConfig.is_peft
```

````

````{py:method} __post_init__()
:canonical: checkpoint.checkpointing.CheckpointingConfig.__post_init__

```{autodoc2-docstring} checkpoint.checkpointing.CheckpointingConfig.__post_init__
```

````

`````

````{py:function} save_model(model: torch.nn.Module, weights_path: str, checkpoint_config: checkpoint.checkpointing.CheckpointingConfig, peft_config: typing.Optional[peft.PeftConfig] = None, tokenizer: typing.Optional[transformers.tokenization_utils.PreTrainedTokenizerBase] = None)
:canonical: checkpoint.checkpointing.save_model

```{autodoc2-docstring} checkpoint.checkpointing.save_model
```
````

````{py:function} load_model(model: torch.nn.Module, weights_path: str, checkpoint_config: checkpoint.checkpointing.CheckpointingConfig)
:canonical: checkpoint.checkpointing.load_model

```{autodoc2-docstring} checkpoint.checkpointing.load_model
```
````

````{py:function} save_optimizer(optimizer: torch.optim.Optimizer, model: torch.nn.Module, weights_path: str, scheduler: typing.Optional[typing.Any] = None)
:canonical: checkpoint.checkpointing.save_optimizer

```{autodoc2-docstring} checkpoint.checkpointing.save_optimizer
```
````

````{py:function} load_optimizer(optimizer: torch.optim.Optimizer, model: torch.nn.Module, weights_path: str, scheduler: typing.Optional[typing.Any] = None)
:canonical: checkpoint.checkpointing.load_optimizer

```{autodoc2-docstring} checkpoint.checkpointing.load_optimizer
```
````

````{py:function} _get_safetensors_index_path(cache_dir: str, repo_id: str) -> str
:canonical: checkpoint.checkpointing._get_safetensors_index_path

```{autodoc2-docstring} checkpoint.checkpointing._get_safetensors_index_path
```
````

````{py:function} _save_peft_adapters(model_state: nemo_automodel.components.checkpoint.stateful_wrappers.ModelState, peft_config: peft.PeftConfig, model_path: str)
:canonical: checkpoint.checkpointing._save_peft_adapters

```{autodoc2-docstring} checkpoint.checkpointing._save_peft_adapters
```
````

````{py:function} _get_hf_peft_config(peft_config: peft.PeftConfig, model_state: nemo_automodel.components.checkpoint.stateful_wrappers.ModelState) -> dict
:canonical: checkpoint.checkpointing._get_hf_peft_config

```{autodoc2-docstring} checkpoint.checkpointing._get_hf_peft_config
```
````

````{py:function} _get_automodel_peft_metadata(peft_config: peft.PeftConfig) -> dict
:canonical: checkpoint.checkpointing._get_automodel_peft_metadata

```{autodoc2-docstring} checkpoint.checkpointing._get_automodel_peft_metadata
```
````

````{py:function} _extract_target_modules(model: torch.nn.Module) -> list[str]
:canonical: checkpoint.checkpointing._extract_target_modules

```{autodoc2-docstring} checkpoint.checkpointing._extract_target_modules
```
````
