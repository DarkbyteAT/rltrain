"""Builder for constructing JAX agents from JSON config dicts.

A config dict with an ``"fqn"`` key resolves to the named class and is
instantiated with the remaining entries as keyword arguments. Sub-configs
nest recursively. Each sub-construction receives a fresh JAX PRNG sub-key
derived from the caller's base key, so Equinox modules that accept ``key=``
get reproducible-but-distinct initialisation.
"""

from __future__ import annotations

import inspect
from types import ModuleType
from typing import Any

import jax
from jaxtyping import PRNGKeyArray

from rltrain.agents.agent import Agent
from rltrain.builders.load import load


def _resolve(cfg: Any, key: PRNGKeyArray) -> Any:
    """Recursively build a JSON config, threading a PRNG key per sub-tree."""
    if isinstance(cfg, dict):
        if "fqn" in cfg:
            fqn = cfg["fqn"]
            try:
                cls = load(fqn)
            except (ModuleNotFoundError, AttributeError) as e:
                raise type(e)(f"Failed to resolve fqn={fqn!r}: {e}") from e
            if isinstance(cls, ModuleType):
                raise TypeError(f"fqn={fqn!r} resolved to a module, expected a class or callable")

            child_keys = jax.random.split(key, max(len(cfg), 1))
            kwargs: dict[str, Any] = {}
            for i, (name, value) in enumerate(cfg.items()):
                if name == "fqn":
                    continue
                kwargs[name] = _resolve(value, child_keys[i])

            if _accepts_key(cls) and "key" not in kwargs:
                kwargs["key"] = key
            return cls(**kwargs)
        return {k: _resolve(v, jax.random.fold_in(key, _key_hash(k))) for k, v in cfg.items()}

    if isinstance(cfg, list):
        child_keys = jax.random.split(key, max(len(cfg), 1))
        return [_resolve(item, child_keys[i]) for i, item in enumerate(cfg)]

    return cfg


def _accepts_key(cls) -> bool:
    """Return True if ``cls`` (or its constructor) takes a ``key`` parameter."""
    try:
        sig = inspect.signature(cls)
    except (ValueError, TypeError):
        return False
    return "key" in sig.parameters


def _key_hash(name: str) -> int:
    """Stable, in-range int derived from a string for ``jax.random.fold_in``."""
    return hash(name) & 0x7FFFFFFF


def agent(fqn: str, *, key: PRNGKeyArray, **kwargs) -> Agent:
    """Build an Equinox-based agent from a JSON config.

    Args:
        fqn: Fully-qualified class name (e.g. ``"rltrain.agents.PPO"``).
        key: Base PRNG key. Sub-keys are derived for each nested sub-module.
        **kwargs: Remaining config fields, possibly nested.

    Returns:
        The instantiated agent module.
    """
    return _resolve({"fqn": fqn, **kwargs}, key)
