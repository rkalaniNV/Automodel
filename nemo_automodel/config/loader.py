import yaml
import importlib
import importlib.util
import os
import sys
from functools import reduce

class ConfigNode:
    """
    A configuration node that wraps a dictionary (or parts of it) from a YAML file.

    This class allows nested dictionaries and lists to be accessed as attributes and
    provides functionality to instantiate objects from configuration.
    """
    def __init__(self, d):
        """
        Initialize the ConfigNode.

        Args:
            d (dict): A dictionary representing configuration options.
        """
        self.__dict__ = {
            k: self._wrap(k, v) for k, v in d.items()
        }

    def _wrap(self, k, v):
        """
        Wrap a configuration value based on its type.

        Args:
            k (str): The key corresponding to the value.
            v: The value to be wrapped.

        Returns:
            The wrapped value.
        """
        if isinstance(v, dict):
            return ConfigNode(v)
        elif isinstance(v, list):
            return [self._wrap('', i) for i in v]
        elif isinstance(v, (int, float)):
            return v
        elif isinstance(v, str) and v.isdigit():
            return int(v)
        else:
            try:
                # try float conversion
                return float(v)
            except Exception:
                # leave strings alone (including _fn and _target_)
                return v

    def instantiate(self, *args, **kwargs):
        """
        Instantiate the target object specified in the configuration.

        This method looks for the "_target_" attribute in the configuration and resolves
        it to a callable function or class which is then instantiated.

        Args:
            *args: Positional arguments for the target instantiation.
            **kwargs: Keyword arguments to override or add to the configuration values.

        Returns:
            The instantiated object.

        Raises:
            AttributeError: If no "_target_" attribute is found in the configuration.
        """
        if not hasattr(self, "_target_"):
            raise AttributeError("No _target_ found to instantiate")
        # resolve the class/function now
        func = self._resolve_target(self._target_)
        # build kwargs
        cfg = {}
        for k, v in self.__dict__.items():
            if k == "_target_":
                continue
            if k.endswith("_fn"):
                # resolve function pointers now
                cfg[k] = self._resolve_target(v)
            else:
                cfg[k] = self._instantiate_value(v)
        # allow user overrides
        cfg.update(kwargs)
        return func(*args, **cfg)

    def _instantiate_value(self, v):
        """
        Recursively instantiate configuration values.

        Args:
            v: The configuration value.

        Returns:
            The instantiated value.
        """
        if isinstance(v, ConfigNode) and hasattr(v, "_target_"):
            return v.instantiate()
        elif isinstance(v, ConfigNode):
            return v.to_dict()
        elif isinstance(v, list):
            return [self._instantiate_value(i) for i in v]
        else:
            return v

    def _resolve_target(self, dotted_path):
        """
        Resolve a dotted path to a Python object.

        This function first attempts a standard import and, if that fails, searches for a
        local module by traversing sys.path.

        Args:
            dotted_path (str): A string representing the dotted path to the object.

        Returns:
            The Python object referenced by the dotted path.

        Raises:
            ImportError: If the target cannot be resolved.
        """
        parts = dotted_path.split(".")

        # try direct import of module and attribute
        if len(parts) > 1:
            try:
                module = importlib.import_module(".".join(parts[:-1]))
                return getattr(module, parts[-1])
            except (ModuleNotFoundError, AttributeError):
                pass

        # try importing first part then attribute chain
        try:
            module = importlib.import_module(parts[0])
            return reduce(getattr, parts[1:], module)
        except (ModuleNotFoundError, AttributeError):
            pass

        # try loading as a local .py file
        for path in sys.path:
            try_path = os.path.join(path, *parts[:-1]) + ".py"
            if os.path.isfile(try_path):
                module_name = "_dynamic_" + "_".join(parts[:-1])
                spec = importlib.util.spec_from_file_location(module_name, try_path)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = mod
                spec.loader.exec_module(mod)
                return getattr(mod, parts[-1])

        raise ImportError(f"Cannot resolve target: {dotted_path}")

    def to_dict(self):
        """
        Convert the configuration node back to a dictionary.

        Returns:
            dict: A dictionary representation of the configuration node.
        """
        return {
            k: self._unwrap(v) for k, v in self.__dict__.items()
        }

    def _unwrap(self, v):
        """
        Recursively convert wrapped configuration values to basic Python types.

        Args:
            v: The configuration value.

        Returns:
            The unwrapped value.
        """
        if isinstance(v, ConfigNode):
            return v.to_dict()
        elif isinstance(v, list):
            return [self._unwrap(i) for i in v]
        else:
            return v

    def get(self, key, default=None):
        """
        Retrieve a configuration value using a dotted key.

        If any component of the path is missing, returns the specified default value.

        Args:
            key (str): The dotted path key.
            default: A default value to return if the key is not found.

        Returns:
            The configuration value or the default value.
        """
        parts = key.split(".")
        current = self
        # TODO(@akoumparouli): reduce?
        for p in parts:
            if isinstance(current, ConfigNode):
                current = current.__dict__.get(p, default)
            elif isinstance(current, list):
                try:
                    current = current[int(p)]
                except:
                    return default
            else:
                return default
            if current is default:
                return default
        return current

    def __contains__(self, key):
        return self.get(key, None) is not None

    def __repr__(self, level=0):
        """
        Return a string representation of the configuration node with indentation.

        Args:
            level (int): The current indentation level.

        Returns:
            str: An indented string representation of the configuration.
        """
        indent = "  " * level
        lines = []
        for k, v in self.__dict__.items():
            if isinstance(v, ConfigNode):
                rep = v.__repr__(level + 1)
            elif isinstance(v, list):
                inner = "\n".join(
                    f"{'  '*(level+1)}{repr(i)}" for i in v
                )
                rep = "[\n" + inner + f"\n{indent}]"
            else:
                rep = repr(v)
            lines.append(f"{indent}{k}: {rep}")
        return "#path:\n" + "\n".join(lines) + f"\n{indent}"

    def __str__(self):
        """
        Return a string representation of the configuration node.

        Returns:
            str: The string representation.
        """
        return self.__repr__(0)

def load_yaml_config(path):
    """
    Load a YAML configuration file and convert it to a ConfigNode.

    Args:
        path (str): The path to the YAML configuration file.

    Returns:
        ConfigNode: A configuration node representing the YAML file.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return ConfigNode(raw)
