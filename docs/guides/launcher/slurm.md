(guides-launcher-slurm)=
# Launch on Slurm

This guide shows how to launch NeMo Automodel jobs on Slurm clusters using the `automodel` CLI and a few YAML settings.

## Prerequisites

- A working Slurm cluster
- Access to an NVIDIA container image or a local environment with Automodel installed

## Minimal Slurm Configuration in YAML

```yaml
slurm:
  job_name: llm-finetune
  nodes: 1
  ntasks_per_node: 8
  time: 00:05:00
  account: <your_account>
  partition: <your_partition>
  container_image: nvcr.io/nvidia/nemo:dev  # or a local .sqsh path
  gpus_per_node: 8
  extra_mounts:
    - /host/path:/container/path
```

## Launch the Job

```bash
automodel llm finetune -c examples/llm/llama_3_2_1b_squad.yaml --nproc-per-node=2
```

## Execute Training from Source Code

If the command is executed inside a Git repository accessible to Slurm workers, the SBATCH script prioritizes the repository source over the container's preinstalled package.

```bash
git clone git@github.com:NVIDIA-NeMo/Automodel.git automodel_test_repo
cd automodel_test_repo/
automodel llm finetune -c examples/llm/llama_3_2_1b_squad.yaml --nproc-per-node=2
```

The job will run using the code in `automodel_test_repo`.

## Next Steps

- {ref}`get-started-quick-start` for first-time setup
- LLM SFT guide for end-to-end config and training

