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

import argparse, dataclasses, subprocess, sys, tempfile
from nemo_automodel.launcher.slurm.template import render_script
from nemo_automodel.launcher.slurm.opts import SlurmConfig

def build_slurm_parser() -> argparse.ArgumentParser:
    """Create an ArgumentParser directly from the dataclass definition."""
    p = argparse.ArgumentParser(description="Create & submit a Slurm script.")
    for f in dataclasses.fields(SlurmConfig):
        cli_name = f"--{f.name.replace('_', '-')}"
        kwargs = dict(help=f.metadata.get("help", ""))
        if f.type is bool:
            # decide whether this flag enables or disables the feature
            kwargs["action"] = "store_true" if f.default is False else "store_false"
        else:
            kwargs["type"] = f.type
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                kwargs["required"] = True
            else:
                # handle both default and default_factory
                default_val = (f.default
                               if f.default is not dataclasses.MISSING
                               else f.default_factory())
                kwargs["default"] = default_val
        p.add_argument(cli_name, **kwargs)
    # dry run
    p.add_argument('--dry-run', action='store_true', help='Do not call sbatch')
    return p

def load_yaml_opts(config_path: str, job_name: str) -> dict:
    """Load and return options from YAML for the given job_name."""
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load config YAML: {e}", file=sys.stderr)
        sys.exit(1)

    if job_name not in cfg:
        print(f"[ERROR] Job name '{job_name}' not found in config file {config_path}", file=sys.stderr)
        print(f"Available sections: {list(cfg.keys())}", file=sys.stderr)
        sys.exit(1)

    return cfg[job_name]

def parse_args_with_config() -> SlurmConfig:
    # Scan sys.argv for --config and --job-name
    config_path, job_name = None, None
    for i, arg in enumerate(sys.argv):
        if arg == "--config":
            assert i + 1 < len(sys.argv), "Please provide --config file.yaml"
            config_path = sys.argv[i + 1]
        elif arg == "--job-name":
            assert i + 1 < len(sys.argv), "Please provide --job-name name-of-job"
            job_name = sys.argv[i + 1]

    yaml_opts = {}
    if config_path:
        if not job_name:
            print("[ERROR] --config requires --job-name to select section in YAML", file=sys.stderr)
            sys.exit(1)
        yaml_opts = load_yaml_opts(config_path, job_name)

    parser = build_slurm_parser()
    cli_args = vars(parser.parse_args())

    # Merge: CLI args override YAML
    return {**yaml_opts, **{k: v for k, v in cli_args.items() if v is not None}}
    # return SlurmConfig(**merged)


def main() -> None:
    args = parse_args_with_config()
    dry_run = args.pop('dry_run', None)
    opts = SlurmConfig(**args)
    script_txt = render_script(opts)

    tmp_path = tempfile.NamedTemporaryFile(
        delete=False, suffix=f"_{opts.job_name}.sbatch", mode="w"
    ).name
    with open(tmp_path, "w") as fh:
        fh.write(script_txt)

    print(f"Generated Slurm script ➜ {tmp_path}")

    if dry_run:
        print("Dry-run: not submitting to Slurm.")
        return

    try:
        out = subprocess.check_output(["sbatch", tmp_path], text=True)
        print(out.strip())
    except subprocess.CalledProcessError as exc:
        print("sbatch submission failed:\n", exc.output)
        sys.exit(exc.returncode)

if __name__ == "__main__":
    main()
