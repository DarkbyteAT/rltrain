from collections.abc import Callable
from functools import partial
from types import ModuleType
from typing import Any


def load(fqn: str) -> ModuleType | type | Callable:
    """Returns a module/class/function from the given fully-qualified name.

    Parameters
    ----------
    `fqn`
        The fully-qualified name of the module/class/function to import.
    """
    parts = fqn.split(".")
    module = ".".join(parts[:-1])
    root = __import__(module)
    for sub in parts[1:]:
        root = getattr(root, sub)
    return root


def resolve(cfg: Any) -> Any:
    """Recursively resolve a JSON config structure into constructed objects.

    Resolution rules:

    - **Plain values** (``str``, ``int``, ``float``, ``bool``, ``None``) pass
      through unchanged.
    - **Dict with** ``"fqn"`` key -- load the class/function via `load`, recurse
      into the remaining values, then construct the object with those kwargs.
      If the dict also contains ``"deferred": true``, wrap in
      `functools.partial` instead of calling immediately.
    - **Dict without** ``"fqn"`` -- recurse into each value.
    - **List** -- recurse into each element.

    Parameters
    ----------
    `cfg`
        A JSON-deserialised config value (dict, list, or scalar).

    Returns
    -------
    The resolved object tree.
    """
    if isinstance(cfg, dict):
        if "fqn" in cfg:
            cls = load(cfg["fqn"])
            deferred = cfg.get("deferred", False)
            kwargs = {k: resolve(v) for k, v in cfg.items() if k not in ("fqn", "deferred")}
            if deferred:
                return partial(cls, **kwargs)
            return cls(**kwargs)
        return {k: resolve(v) for k, v in cfg.items()}

    if isinstance(cfg, list):
        return [resolve(item) for item in cfg]

    return cfg
