# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Example launcher for knowledge distillation fine-tuning.

Usage (single GPU):
    python examples/llm/knowledge_distillation.py -c examples/llm/llama_3_2_1b_kd.yaml

When run without ``-c`` it defaults to the YAML above.
"""

from __future__ import annotations

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.llm.knowledge_distillation import (
    KnowledgeDistillationRecipeForNextTokenPrediction,
)


def main(default_config_path="examples/llm/llama_3_2_1b_kd.yaml") -> None:
    """Entry-point mirroring ``examples/llm/finetune.py`` but for KD."""
    cfg = parse_args_and_load_config(default_config_path)
    recipe = KnowledgeDistillationRecipeForNextTokenPrediction(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":  # pragma: no cover
    main() 