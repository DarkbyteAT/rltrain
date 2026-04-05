from collections.abc import Callable

import gymnasium as gym
import gymnasium.vector as vgym

from rltrain.utils.builders.load import load


def wrap(fqn: str, env_fn: Callable[[], gym.Env], **kwargs) -> Callable[[], gym.Env]:
    wrapper_type = load(fqn)
    return lambda: wrapper_type(env_fn(), **kwargs)


def _make_env_fn(id: str, wrappers: list[dict[str]], **kwargs) -> Callable[[], gym.Env]:
    """Build a factory that creates a wrapped gymnasium env."""

    def env_fn() -> gym.Env:
        return gym.make(id, **kwargs)

    for wrapper in wrappers:
        env_fn = wrap(env_fn=env_fn, **wrapper)
    return env_fn


def env(id: str, wrappers: list[dict[str]], **kwargs) -> vgym.VectorEnv:
    return vgym.SyncVectorEnv([_make_env_fn(id, wrappers, **kwargs)])


def eval_env(id: str, wrappers: list[dict[str]], **kwargs) -> gym.Env:
    """Build a single wrapped env with ``render_mode="rgb_array"`` for evaluation."""
    kwargs.setdefault("render_mode", "rgb_array")
    return _make_env_fn(id, wrappers, **kwargs)()
