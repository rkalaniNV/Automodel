"""
Utilities to record nn.Module constructor arguments at instantiation time.

This module provides a scoped context manager that uses Python's profiling
hooks to capture arguments passed to `__init__` for subclasses of
`torch.nn.Module`. The captured arguments are stored on the instance as:

    - _automodel_init_args: tuple of var-positional args (if present)
    - _automodel_init_kwargs: dict of keyword-like arguments (including named
      parameters and var-keyword **kwargs)

The context is intended to be enabled during model construction only.
"""

from __future__ import annotations

import builtins
import importlib
import inspect
import sys
from types import FrameType
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class InitArgTraceRecorder:
    """
    Scoped recorder that attaches init-time args/kwargs to nn.Module instances.

    Example
    -------
    >>> with InitArgTraceRecorder(module_prefixes=("transformers", "nemo_automodel")):
    ...     model = AutoModelForCausalLM.from_pretrained("gpt2")
    ...     # All submodules constructed during this block will have
    ...     # `_automodel_init_kwargs` recorded when possible.
    """

    def __init__(self, module_prefixes: Tuple[str, ...] = ("transformers", "nemo_automodel")) -> None:
        self.module_prefixes = module_prefixes
        self._prev_tracer = None
        self._patched_inits: Dict[type, Any] = {}
        self._orig_import_module = None
        self._orig___import__ = None

    def _should_record(self, cls: type) -> bool:
        try:
            mod = cls.__module__
        except Exception:
            return False
        return any(mod.startswith(prefix) for prefix in self.module_prefixes)

    # --- Monkey-patch based recording (robust and fast) ---
    def _sanitize_value(self, v: Any) -> Any:
        if isinstance(v, torch.Tensor) and v.ndim == 0:
            try:
                return v.item()
            except Exception:
                return v
        return v

    def _patch_class_init(self, cls: type) -> None:
        if cls in self._patched_inits:
            return
        try:
            original_init = cls.__init__
        except AttributeError:
            return
        if not callable(original_init):
            return
        if original_init is nn.Module.__init__:
            return
        if not self._should_record(cls):
            return

        sig = None
        try:
            sig = inspect.signature(original_init)
        except Exception:
            pass

        def wrapped(self, *args, **kwargs):
            # Bind and store before calling original __init__
            bound_kwargs: Dict[str, Any] = {}
            varargs_tuple: Tuple[Any, ...] = args
            if sig is not None:
                try:
                    bound = sig.bind_partial(self, *args, **kwargs)
                    bound.apply_defaults()
                    # Exclude self
                    for k, v in bound.arguments.items():
                        if k == "self":
                            continue
                        bound_kwargs[k] = self._sanitize_value(v)
                    # Capture *args if present in signature
                    for name, param in sig.parameters.items():
                        if param.kind == inspect.Parameter.VAR_POSITIONAL and name in bound.arguments:
                            varargs_tuple = bound.arguments[name]
                            break
                except Exception:
                    # fallback: raw kwargs
                    bound_kwargs = {k: self._sanitize_value(v) for k, v in kwargs.items()}
            else:
                bound_kwargs = {k: self._sanitize_value(v) for k, v in kwargs.items()}

            try:
                self._automodel_init_args = tuple(varargs_tuple) if isinstance(varargs_tuple, tuple) else ()
            except Exception:
                pass
            try:
                self._automodel_init_kwargs = dict(bound_kwargs)
            except Exception:
                pass
            return original_init(self, *args, **kwargs)

        try:
            self._patched_inits[cls] = original_init
            setattr(cls, "__init__", wrapped)
        except Exception:
            self._patched_inits.pop(cls, None)

    def _patch_module(self, module: Any) -> None:
        mod_name = getattr(module, "__name__", "")
        if not any(mod_name.startswith(prefix) for prefix in self.module_prefixes):
            return
        for attr in list(getattr(module, "__dict__", {}).values()):
            if isinstance(attr, type) and issubclass(attr, nn.Module):
                self._patch_class_init(attr)

    def _patch_existing_modules(self) -> None:
        for name, module in list(sys.modules.items()):
            if not isinstance(name, str) or module is None:
                continue
            if any(name.startswith(prefix) for prefix in self.module_prefixes):
                self._patch_module(module)

    def _wrap_import_module(self):
        self._orig_import_module = importlib.import_module

        def _import_module(name, package=None):
            module = self._orig_import_module(name, package)
            # Patch the module and any immediate submodules that were loaded
            self._patch_module(module)
            for mod_name, mod in list(sys.modules.items()):
                if isinstance(mod_name, str) and mod_name.startswith(name):
                    self._patch_module(mod)
            return module

        importlib.import_module = _import_module

    def _wrap___import__(self):
        self._orig___import__ = builtins.__import__

        def _import(name, globals=None, locals=None, fromlist=(), level=0):
            module = self._orig___import__(name, globals, locals, fromlist, level)
            # Patch the module and any submodules loaded as a side-effect
            for mod_name, mod in list(sys.modules.items()):
                if isinstance(mod_name, str) and (mod_name == name or mod_name.startswith(name + ".")):
                    self._patch_module(mod)
            return module

        builtins.__import__ = _import

    def _global_tracer(self, frame: FrameType, event: str, arg: Any):
        if event != "call":
            return None
        if frame.f_code.co_name != "__init__":
            return None

        self_obj = frame.f_locals.get("self")
        if self_obj is None or not isinstance(self_obj, nn.Module):
            return None
        cls = self_obj.__class__
        if not self._should_record(cls):
            return None

        captured = {"done": False}

        def _local_tracer(local_frame: FrameType, local_event: str, local_arg: Any):
            if captured["done"]:
                return None
            if local_event not in ("line", "return"):
                return _local_tracer
            # At first executable line (or at return), locals are bound
            try:
                sig = inspect.signature(cls.__init__)
            except (TypeError, ValueError):
                captured["done"] = True
                return None

            local_vars = local_frame.f_locals
            kwargs: Dict[str, Any] = {}
            varargs_name = None
            varkw_name = None
            for name, param in sig.parameters.items():
                if name == "self":
                    continue
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    varargs_name = name
                elif param.kind == inspect.Parameter.VAR_KEYWORD:
                    varkw_name = name
                else:
                    if name in local_vars:
                        kwargs[name] = local_vars[name]

            # Merge **kwargs if present
            if varkw_name and varkw_name in local_vars and isinstance(local_vars[varkw_name], dict):
                try:
                    kwargs.update(dict(local_vars[varkw_name]))
                except Exception:
                    pass

            # Extract *args tuple if present
            args_tuple: Tuple[Any, ...] = ()
            if varargs_name and varargs_name in local_vars and isinstance(local_vars[varargs_name], tuple):
                args_tuple = local_vars[varargs_name]

            # Sanitize zero-dim tensors -> Python scalars for portability
            sanitized_kwargs: Dict[str, Any] = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor) and v.ndim == 0:
                    try:
                        v = v.item()
                    except Exception:
                        pass
                sanitized_kwargs[k] = v

            try:
                self_obj._automodel_init_args = args_tuple
            except Exception:
                pass
            try:
                self_obj._automodel_init_kwargs = sanitized_kwargs
            except Exception:
                pass

            captured["done"] = True
            return None

        return _local_tracer

    def __enter__(self):
        # 1) Patch already-loaded modules and set import hooks for modules loaded later
        self._patch_existing_modules()
        self._wrap_import_module()
        self._wrap___import__()
        # 2) Also enable a tracer as a fallback for cases where patching is missed
        self._prev_tracer = sys.gettrace()
        sys.settrace(self._global_tracer)
        return self

    def __exit__(self, exc_type, exc, tb):
        # Restore tracer
        sys.settrace(self._prev_tracer)
        self._prev_tracer = None
        # Restore import hooks
        if self._orig_import_module is not None:
            importlib.import_module = self._orig_import_module
            self._orig_import_module = None
        if self._orig___import__ is not None:
            builtins.__import__ = self._orig___import__
            self._orig___import__ = None
        # Restore patched __init__ for all classes
        for cls, orig in list(self._patched_inits.items()):
            try:
                setattr(cls, "__init__", orig)
            except Exception:
                pass
        self._patched_inits.clear()
        return False
