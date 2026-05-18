"""FQN loader and recursive config resolver for the builder system."""

from collections.abc import Callable
from functools import partial
from types import ModuleType
from typing import Any


def load(fqn: str) -> ModuleType | type | Callable[..., Any]:
    """Return the module/class/function at the given fully-qualified name."""
    parts = fqn.split(".")
    module = ".".join(parts[:-1])
    root = __import__(module)
    for sub in parts[1:]:
        root = getattr(root, sub)
    return root


def resolve(cfg: Any) -> Any:
    """Recursively resolve a JSON config tree into constructed objects.

    Dict with ``"fqn"`` key → load + instantiate; with ``"deferred": true``
    → ``functools.partial`` instead. Plain values pass through.
    """
    if isinstance(cfg, dict):
        if "fqn" in cfg:
            fqn = cfg["fqn"]
            try:
                cls = load(fqn)
            except (ModuleNotFoundError, AttributeError) as e:
                raise type(e)(f"Failed to resolve fqn={fqn!r}: {e}") from e
            if isinstance(cls, ModuleType):
                raise TypeError(f"fqn={fqn!r} resolved to a module, expected a class or callable")
            deferred = cfg.get("deferred", False)
            kwargs = {k: resolve(v) for k, v in cfg.items() if k not in ("fqn", "deferred")}
            if deferred:
                return partial(cls, **kwargs)
            return cls(**kwargs)
        return {k: resolve(v) for k, v in cfg.items()}

    if isinstance(cfg, list):
        return [resolve(item) for item in cfg]

    return cfg
