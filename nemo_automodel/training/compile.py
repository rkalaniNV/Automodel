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

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CompileConfig:
    """Configuration for torch.compile settings."""

    def __init__(
        self,
        enabled: bool = False,
        mode: str = "default",
        fullgraph: bool = False,
        dynamic: bool = True,
        backend: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """Initialize compile configuration.

        Args:
            enabled: Whether to enable torch.compile.
            mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune').
            fullgraph: Whether to compile the entire graph.
            dynamic: Whether to enable dynamic shapes.
            backend: Backend to use for compilation. If None, uses TORCH_COMPILE_BACKEND env var or "inductor".
            options: Additional options to pass to torch.compile.
        """
        self.enabled = enabled
        self.mode = mode
        self.fullgraph = fullgraph
        self.dynamic = dynamic
        self.backend = backend if backend is not None else os.environ.get("TORCH_COMPILE_BACKEND", "inductor")
        self.options = options or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "fullgraph": self.fullgraph,
            "dynamic": self.dynamic,
            "backend": self.backend,
            "options": self.options,
        }


def compile_model(model: nn.Module, config: CompileConfig) -> nn.Module:
    """Compile the model if enabled, following TorchTune pattern.
    
    Args:
        model: The model to compile.
        config: Compile configuration.
        
    Returns:
        The compiled model or original model if compilation is disabled.
    """
    if not config.enabled:
        logger.info("torch.compile is disabled")
        return model

    try:
        logger.info(f"Selectively compiling transformer layers with mode: {config.mode}")
        
        # Convert options to dictionary if it's a ConfigNode
        options_dict = config.options.to_dict() if hasattr(config.options, 'to_dict') else dict(config.options)
        
        # Prepare torch.compile arguments, excluding None values
        compile_kwargs = {
            "mode": config.mode,
            "fullgraph": config.fullgraph,
            "dynamic": config.dynamic,
        }
        if config.backend is not None:
            compile_kwargs["backend"] = config.backend
        compile_kwargs.update(options_dict)
        
        # Find and compile transformer layers
        compiled_layers = 0
        
        # Look for common transformer layer patterns in LLM models
        for name, module in model.named_modules():
            # Skip if this is not a leaf module (has children)
            if list(module.children()):
                continue
                
            # Look for specific transformer layer patterns
            if any(pattern in name.lower() for pattern in [
                'transformer.layer',  # Llama, GPT-2 style
                'transformer_block',  # Generic transformer blocks
                'block',  # Generic blocks
                'layer.',  # Layer prefix
                'decoder_layer',  # Decoder layers
                'encoder_layer',  # Encoder layers
            ]):
                if isinstance(module, nn.Module) and hasattr(module, 'forward'):
                    try:
                        compiled_module = torch.compile(
                            module,
                            **compile_kwargs,
                        )
                        
                        # Replace the module in the parent
                        parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else ''
                        if parent_name:
                            parent = model.get_submodule(parent_name)
                            child_name = name.split('.')[-1]
                            setattr(parent, child_name, compiled_module)
                        else:
                            # Root level module
                            setattr(model, name, compiled_module)
                        
                        compiled_layers += 1
                        logger.info(f"Compiled transformer layer: {name}")
                    except Exception as e:
                        logger.warning(f"Failed to compile layer {name}: {e}")
                        continue
        
        if compiled_layers > 0:
            logger.info(f"Successfully compiled {compiled_layers} transformer layers")
        else:
            logger.warning("No transformer layers found to compile - falling back to full model compilation")
            # Fallback to full model compilation if no layers found
            try:
                compiled_model = torch.compile(
                    model,
                    **compile_kwargs,
                )
                logger.info("Full model compilation successful")
                return compiled_model
            except Exception as e:
                logger.error(f"Full model compilation also failed: {e}")
                raise RuntimeError(f"Compilation failed: {e}") from e
        
        return model
    except Exception as e:
        logger.error(f"Selective compilation failed: {e}")
        raise RuntimeError(f"Selective compilation failed: {e}") from e


def create_compile_config_from_dict(config_dict: Dict[str, Any]) -> CompileConfig:
    """Create a CompileConfig from a dictionary.

    Args:
        config_dict: Dictionary containing compile configuration.

    Returns:
        CompileConfig instance.
    """
    return CompileConfig(
        enabled=config_dict.get("enabled", False),
        mode=config_dict.get("mode", "default"),
        fullgraph=config_dict.get("fullgraph", False),
        dynamic=config_dict.get("dynamic", True),
        backend=config_dict.get("backend", None),
        options=config_dict.get("options", {}),
    )


def build_compile_config(cfg: Optional[Dict[str, Any]]) -> CompileConfig:
    """Build a compile config from configuration.

    Args:
        cfg: Configuration dictionary for compilation.

    Returns:
        CompileConfig instance.
    """
    if cfg is None:
        return CompileConfig(enabled=False)
    else:
        return create_compile_config_from_dict(cfg) 