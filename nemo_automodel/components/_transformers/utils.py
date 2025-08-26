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

from collections import defaultdict
from typing import Any

from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForTextToWaveform,
)

# an automodel factory for loading the huggingface models from correct class
AUTOMODEL_FACTORY = defaultdict(lambda: AutoModelForCausalLM)
AUTOMODEL_FACTORY["qwen2_5_vl"] = AutoModelForImageTextToText
AUTOMODEL_FACTORY["qwen2_vl"] = AutoModelForImageTextToText
AUTOMODEL_FACTORY["qwen2_5_omni"] = AutoModelForTextToWaveform
AUTOMODEL_FACTORY["llava"] = AutoModelForImageTextToText
AUTOMODEL_FACTORY["internvl"] = AutoModelForImageTextToText
AUTOMODEL_FACTORY["gemma3"] = AutoModelForImageTextToText
AUTOMODEL_FACTORY["smolvlm"] = AutoModelForImageTextToText
AUTOMODEL_FACTORY["mistral3"] = AutoModelForImageTextToText
AUTOMODEL_FACTORY["llama4"] = AutoModelForImageTextToText


def resolve_model_class(model_name: str) -> nn.Module:
    if model_name.lower() in AUTOMODEL_FACTORY.keys():
        return AUTOMODEL_FACTORY[model_name.lower()]
    return AutoModelForCausalLM


def sliding_window_overwrite(model_name: str) -> dict[str, Any]:
    """Returns configuration overrides to handle sliding window settings based on model rules.

    Args:
        model_name: The HuggingFace model name or path to load configuration from

    Returns:
        dict: Dictionary with overwrite values, or empty dict if no overwrites needed
    """
    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    overwrite_dict = {}

    # Override sliding_window setting to address a HF mismatch relevant to use_sliding_window
    # TODO(@zhiyul): remove this once the bug is fixed https://github.com/huggingface/transformers/issues/38002
    if hasattr(hf_config, "use_sliding_window") and hf_config.use_sliding_window == False:
        assert hasattr(hf_config, "sliding_window")
        overwrite_dict = {
            "sliding_window": None,
        }
        print(f"use_sliding_window=False in config - overriding sliding_window parameter to None: {overwrite_dict}")

    return overwrite_dict
