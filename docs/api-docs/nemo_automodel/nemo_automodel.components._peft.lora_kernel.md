# {py:mod}`nemo_automodel.components._peft.lora_kernel`

```{py:module} nemo_automodel.components._peft.lora_kernel
```

```{autodoc2-docstring} nemo_automodel.components._peft.lora_kernel
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`forward_autotune_configs <nemo_automodel.components._peft.lora_kernel.forward_autotune_configs>`
  - ```{autodoc2-docstring} nemo_automodel.components._peft.lora_kernel.forward_autotune_configs
    :summary:
    ```
* - {py:obj}`get_pid_coords <nemo_automodel.components._peft.lora_kernel.get_pid_coords>`
  - ```{autodoc2-docstring} nemo_automodel.components._peft.lora_kernel.get_pid_coords
    :summary:
    ```
* - {py:obj}`inner_kernel <nemo_automodel.components._peft.lora_kernel.inner_kernel>`
  - ```{autodoc2-docstring} nemo_automodel.components._peft.lora_kernel.inner_kernel
    :summary:
    ```
* - {py:obj}`block_vector_mul <nemo_automodel.components._peft.lora_kernel.block_vector_mul>`
  - ```{autodoc2-docstring} nemo_automodel.components._peft.lora_kernel.block_vector_mul
    :summary:
    ```
* - {py:obj}`lora_forward_kernel <nemo_automodel.components._peft.lora_kernel.lora_forward_kernel>`
  - ```{autodoc2-docstring} nemo_automodel.components._peft.lora_kernel.lora_forward_kernel
    :summary:
    ```
* - {py:obj}`lora_forward_wrapper <nemo_automodel.components._peft.lora_kernel.lora_forward_wrapper>`
  - ```{autodoc2-docstring} nemo_automodel.components._peft.lora_kernel.lora_forward_wrapper
    :summary:
    ```
* - {py:obj}`da_dx_autotune_configs <nemo_automodel.components._peft.lora_kernel.da_dx_autotune_configs>`
  - ```{autodoc2-docstring} nemo_automodel.components._peft.lora_kernel.da_dx_autotune_configs
    :summary:
    ```
* - {py:obj}`lora_da_dx_kernel <nemo_automodel.components._peft.lora_kernel.lora_da_dx_kernel>`
  - ```{autodoc2-docstring} nemo_automodel.components._peft.lora_kernel.lora_da_dx_kernel
    :summary:
    ```
* - {py:obj}`lora_da_dx_update_wrapper <nemo_automodel.components._peft.lora_kernel.lora_da_dx_update_wrapper>`
  - ```{autodoc2-docstring} nemo_automodel.components._peft.lora_kernel.lora_da_dx_update_wrapper
    :summary:
    ```
* - {py:obj}`db_autotune_configs <nemo_automodel.components._peft.lora_kernel.db_autotune_configs>`
  - ```{autodoc2-docstring} nemo_automodel.components._peft.lora_kernel.db_autotune_configs
    :summary:
    ```
* - {py:obj}`lora_db_kernel <nemo_automodel.components._peft.lora_kernel.lora_db_kernel>`
  - ```{autodoc2-docstring} nemo_automodel.components._peft.lora_kernel.lora_db_kernel
    :summary:
    ```
* - {py:obj}`lora_db_update_wrapper <nemo_automodel.components._peft.lora_kernel.lora_db_update_wrapper>`
  - ```{autodoc2-docstring} nemo_automodel.components._peft.lora_kernel.lora_db_update_wrapper
    :summary:
    ```
````

### API

````{py:function} forward_autotune_configs()
:canonical: nemo_automodel.components._peft.lora_kernel.forward_autotune_configs

```{autodoc2-docstring} nemo_automodel.components._peft.lora_kernel.forward_autotune_configs
```
````

````{py:function} get_pid_coords(M, N, BLOCK_SIZE_M: triton.language.constexpr, BLOCK_SIZE_N: triton.language.constexpr, GROUP_SIZE_M: triton.language.constexpr)
:canonical: nemo_automodel.components._peft.lora_kernel.get_pid_coords

```{autodoc2-docstring} nemo_automodel.components._peft.lora_kernel.get_pid_coords
```
````

````{py:function} inner_kernel(pid_m, pid_n, a_ptr, b_ptr, M, K, N, stride_am, stride_ak, stride_bk, stride_bn, BLOCK_SIZE_M: triton.language.constexpr, BLOCK_SIZE_K: triton.language.constexpr, BLOCK_SIZE_N: triton.language.constexpr, scale)
:canonical: nemo_automodel.components._peft.lora_kernel.inner_kernel

```{autodoc2-docstring} nemo_automodel.components._peft.lora_kernel.inner_kernel
```
````

````{py:function} block_vector_mul(pid_m, pid_n, ab_result, c_ptr, d_ptr, M, N, L, stride_cn, stride_cl, stride_dm, stride_dl, BLOCK_SIZE_M: triton.language.constexpr, BLOCK_SIZE_N: triton.language.constexpr, BLOCK_SIZE_L: triton.language.constexpr)
:canonical: nemo_automodel.components._peft.lora_kernel.block_vector_mul

```{autodoc2-docstring} nemo_automodel.components._peft.lora_kernel.block_vector_mul
```
````

````{py:function} lora_forward_kernel(x_ptr, la_ptr, lb_ptr, res_ptr, M, N, K, L, stride_x_m, stride_x_k, stride_la_k, stride_la_n, stride_lb_n, stride_lb_l, stride_res_m, stride_res_l, scale, BLOCK_SIZE_M: triton.language.constexpr, BLOCK_SIZE_N: triton.language.constexpr, BLOCK_SIZE_K: triton.language.constexpr, BLOCK_SIZE_L: triton.language.constexpr, GROUP_SIZE_M: triton.language.constexpr)
:canonical: nemo_automodel.components._peft.lora_kernel.lora_forward_kernel

```{autodoc2-docstring} nemo_automodel.components._peft.lora_kernel.lora_forward_kernel
```
````

````{py:function} lora_forward_wrapper(x, lora_A, lora_B, res, scale, dtype=torch.float32)
:canonical: nemo_automodel.components._peft.lora_kernel.lora_forward_wrapper

```{autodoc2-docstring} nemo_automodel.components._peft.lora_kernel.lora_forward_wrapper
```
````

````{py:function} da_dx_autotune_configs()
:canonical: nemo_automodel.components._peft.lora_kernel.da_dx_autotune_configs

```{autodoc2-docstring} nemo_automodel.components._peft.lora_kernel.da_dx_autotune_configs
```
````

````{py:function} lora_da_dx_kernel(dy_ptr, b_ptr, a_ptr, dx_ptr, dyb_ptr, M, K, N, L, stride_dy_m, stride_dy_k, stride_lorab_k, stride_lorab_n, stride_loraa_n, stride_loraa_l, stride_dx_m, stride_dx_l, stride_dyb_m, stride_dyb_n, scale, BLOCK_SIZE_M: triton.language.constexpr, GROUP_SIZE_M: triton.language.constexpr, BLOCK_SIZE_N: triton.language.constexpr, BLOCK_SIZE_K: triton.language.constexpr, BLOCK_SIZE_L: triton.language.constexpr)
:canonical: nemo_automodel.components._peft.lora_kernel.lora_da_dx_kernel

```{autodoc2-docstring} nemo_automodel.components._peft.lora_kernel.lora_da_dx_kernel
```
````

````{py:function} lora_da_dx_update_wrapper(xt, dy, lora_B, lora_A, scale, dtype=torch.float32)
:canonical: nemo_automodel.components._peft.lora_kernel.lora_da_dx_update_wrapper

```{autodoc2-docstring} nemo_automodel.components._peft.lora_kernel.lora_da_dx_update_wrapper
```
````

````{py:function} db_autotune_configs()
:canonical: nemo_automodel.components._peft.lora_kernel.db_autotune_configs

```{autodoc2-docstring} nemo_automodel.components._peft.lora_kernel.db_autotune_configs
```
````

````{py:function} lora_db_kernel(a_ptr, b_ptr, c_ptr, M, K, N, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, scale, BLOCK_SIZE_M: triton.language.constexpr, BLOCK_SIZE_K: triton.language.constexpr, BLOCK_SIZE_N: triton.language.constexpr, GROUP_SIZE_M: triton.language.constexpr)
:canonical: nemo_automodel.components._peft.lora_kernel.lora_db_kernel

```{autodoc2-docstring} nemo_automodel.components._peft.lora_kernel.lora_db_kernel
```
````

````{py:function} lora_db_update_wrapper(lora_A, xt, dy, scale, dtype=torch.float32)
:canonical: nemo_automodel.components._peft.lora_kernel.lora_db_update_wrapper

```{autodoc2-docstring} nemo_automodel.components._peft.lora_kernel.lora_db_update_wrapper
```
````
