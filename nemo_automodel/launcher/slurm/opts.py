# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/usr/bin/env python3
"""
make_and_submit.py  ·  Generate a Slurm batch script from CLI args (defined in a
dataclass), drop it to a temp file, and optionally submit it via sbatch.

Example
-------
python make_and_submit.py \
  --job-name llama3-test \
  --nodes 2 \
  --time 00:05:00 \
  --command "pip3 install torchdata; python3 /lustre/.../finetune.py --config cfg.yaml" \
  --dry-run          # inspect only
"""

import os
from dataclasses import dataclass, field

@dataclass
class SlurmConfig:
    # Slurm basics
    job_name: str = field(
        metadata=dict(help="Job name for Slurm (-J)")
    )
    nodes: int = field(
        default=1,
        metadata=dict(help="Number of nodes (-N)")
    )
    ntasks: int = field(
        default=8,
        metadata=dict(help="ntasks per node (--ntasks-per-node)")
    )
    time: str = field(
        default="00:05:00",
        metadata=dict(help="Wall-clock time limit")
    )
    account: str = field(
        default="coreai_dlalgo_llm",
        metadata=dict(help="Slurm account (-A)")
    )
    partition: str = field(
        default="batch",
        metadata=dict(help="Partition/queue (-p)")
    )

    # Container / mounts
    container: str = field(
        default="/lustre/fsw/coreai_dlalgo_llm/akoumparouli/sqsh_images/"
                "nemo:25.04.rc4.unsloth.sqsh",
        metadata=dict(help="SquashFS / OCI image path")
    )
    nemo_mount: str = field(
        default="/lustre/fsw/coreai_dlalgo_llm/akoumparouli/Automodel",
        metadata=dict(help="Host directory to mount inside container")
    )
    nemo_target: str = field(
        default="/opt/Automodel",
        metadata=dict(help="Container mount target for project")
    )
    cache_mount: str = field(
        default="/lustre/fsw/coreai_dlalgo_llm/akoumparouli/cp/.cache",
        metadata=dict(help="Host HF cache directory")
    )
    cache_target: str = field(
        default="/root/.cache",
        metadata=dict(help="Cache target inside container")
    )
    extra_mounts: str = field(
        default="/lustre/fsw/coreai_dlalgo_llm/akoumparouli:"
                "/lustre/fsw/coreai_dlalgo_llm/akoumparouli",
        metadata=dict(help="Additional mounts host:container (comma-separated)"))
    log_dir: str = field(
        default_factory=lambda: os.path.join(os.getcwd(), "logs"),
        metadata=dict(help="Directory for slurm stdout file")
    )

    # Misc env / training specifics
    master_port: int = field(
        default=13742,
        metadata=dict(help="Port for multinode")
    )
    gpus_per_node: int = field(
        default=8,
        metadata=dict(help="GPUs per node")
    )
    wandb_key: str = field(
        default=os.environ.get('WANDB_API_KEY', ''),
        metadata=dict(help="W&B key or env reference")
    )
    hf_token: str = field(
        default=os.environ.get('HF_TOKEN', ''),
        metadata=dict(help="HF-TOKEN key to use for retrieving gated assets from HuggingFace Hub.")
    )
    hf_home: str = field(
        default="/root/.cache/huggingface",
        metadata=dict(help="HF_HOME inside container")
    )

    # User command
    command: str = field(
        default='',
        metadata=dict(help="Shell command(s) to run inside container")
    )
