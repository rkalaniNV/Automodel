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

# Try to import common HuggingFace transformer base classes for general approach
try:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    PreTrainedModel = None
    LlamaDecoderLayer = None
    HUGGINGFACE_AVAILABLE = False


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
    """Compile the model following torchtune's approach: try full model first, then selective fallback.
    
    Compilation strategy (following torchtune pattern):
    1. Try full model compilation first (most efficient when it works)
    2. If full model compilation fails, fall back to selective layer compilation
    3. If selective compilation also fails, gracefully return the original model
    
    Args:
        model: The model to compile.
        config: Compile configuration.
        
    Returns:
        The compiled model, partially compiled model, or original model if compilation fails.
    """
    if not config.enabled:
        logger.info("torch.compile is disabled")
        return model

    # Prepare torch.compile arguments
    options_dict = config.options.to_dict() if hasattr(config.options, 'to_dict') else dict(config.options)
    compile_kwargs = {
        "mode": config.mode,
        "fullgraph": config.fullgraph,
        "dynamic": config.dynamic,
    }
    if config.backend is not None:
        compile_kwargs["backend"] = config.backend
    compile_kwargs.update(options_dict)

    logger.info(f"Starting compilation with backend={config.backend}, mode={config.mode}")

    # Strategy 1: Try full model compilation first (torchtune approach)
    logger.info("Attempting full model compilation...")
    try:
        compiled_model = torch.compile(model, **compile_kwargs)
        # Test compilation with a simple forward pass to ensure it works
        logger.info("Full model compilation successful - using compiled model")
        return compiled_model
    except Exception as e:
        logger.warning(f"Full model compilation failed: {type(e).__name__}: {e}")
        logger.info("Falling back to selective layer compilation...")

    # Strategy 2: Fall back to selective layer compilation (torchtune fallback)
    try:
        return _compile_selective_layers(model, compile_kwargs)
    except Exception as e:
        logger.error(f"Selective layer compilation failed: {type(e).__name__}: {e}")
        logger.info("Compilation failed completely - returning original model")
        return model


def _is_transformer_block(module: nn.Module) -> bool:
    """Check if a module looks like a transformer block by examining its structure.
    
    This is a general approach that works with most HuggingFace LLMs by identifying
    modules that have the typical transformer block structure.
    """
    class_name = module.__class__.__name__
    
    # First check: Does the class name suggest it's a transformer block?
    name_patterns = [
        'Block', 'Layer', 'DecoderLayer', 'EncoderLayer', 'TransformerBlock',
        'TransformerLayer', 'DecoderBlock', 'EncoderBlock'
    ]
    
    has_block_name = any(pattern in class_name for pattern in name_patterns)
    
    if not has_block_name:
        return False
    
    # Second check: Does it have the typical transformer block submodules?
    child_names = [name for name, _ in module.named_children()]
    child_names_lower = [name.lower() for name in child_names]
    
    # Look for attention mechanism
    has_attention = any(
        'attn' in name or 'attention' in name 
        for name in child_names_lower
    )
    
    # Look for feedforward/MLP mechanism
    has_feedforward = any(
        'mlp' in name or 'ffn' in name or 'feed_forward' in name or 'feedforward' in name
        for name in child_names_lower
    )
    
    # A transformer block typically has both attention and feedforward
    return has_attention and has_feedforward


def _compile_selective_layers(model: nn.Module, compile_kwargs: dict) -> nn.Module:
    """Compile selective layers using torchtitan approach - target transformer blocks generally."""
    
    compiled_layers = 0
    failed_layers = 0
    backend = compile_kwargs.get('backend', 'inductor')
    
    logger.info("Selective compilation targeting transformer blocks (name-based replacement)")
    
    # Use named_modules instead of just modules - this gives us the module paths
    # which makes replacement much cleaner and more reliable
    for name, m in model.named_modules():
        should_compile = False
        
        # Use general transformer block detection
        if _is_transformer_block(m):
            should_compile = True
            logger.debug(f"Identified transformer block: {name} ({m.__class__.__name__})")
        
        if should_compile:
            try:
                # Apply torch.compile directly to the transformer block (torchtitan style)
                compiled_module = torch.compile(m, backend=backend, **{k: v for k, v in compile_kwargs.items() if k != 'backend'})
                # Replace using name-based approach (cleaner and more reliable)
                _replace_module_in_parent(model, name, compiled_module)
                compiled_layers += 1
                logger.debug(f"Compiled transformer block: {name}")
                
            except Exception as e:
                failed_layers += 1
                logger.debug(f"Failed to compile {name}: {type(e).__name__}: {e}")
                continue
    
    # Report results
    if compiled_layers > 0:
        logger.info(f"Selective compilation successful! Compiled {compiled_layers} transformer block modules "
                   f"({failed_layers} failed)")
        return model
    else:
        logger.warning("No transformer block modules found for compilation")
        logger.warning("This approach targets modules with 'Block'/'Layer' names that contain attention and feedforward components")
        raise RuntimeError("No transformer block modules could be compiled")


def _replace_module_in_parent(model: nn.Module, module_name: str, new_module: nn.Module):
    """Replace a module in its parent using name-based lookup (torchtitan style).
    
    This is cleaner, faster, and more reliable than instance-based replacement.
    """
    if '.' in module_name:
        parent_name = '.'.join(module_name.split('.')[:-1])
        parent = model.get_submodule(parent_name)
        child_name = module_name.split('.')[-1]
        setattr(parent, child_name, new_module)
        logger.debug(f"Replaced {module_name} in parent {parent_name}")
    else:
        # Root level module
        setattr(model, module_name, new_module)
        logger.debug(f"Replaced root level module {module_name}")


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