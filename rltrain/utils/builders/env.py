from collections.abc import Callable

import gymnasium as gym
import gymnasium.vector as vgym

from rltrain.utils.builders import load


def wrap(fqn: str, env_fn: Callable[[], gym.Env], **kwargs) -> Callable[[], gym.Env]:
    wrapper_type = load(fqn)
    return lambda: wrapper_type(env_fn(), **kwargs)


def env(id: str, wrappers: list[dict[str]], **kwargs) -> vgym.VectorEnv:
    def env_fn() -> gym.Env:
        return gym.make(id, **kwargs)

    for wrapper in wrappers:
        env_fn = wrap(env_fn=env_fn, **wrapper)
    return vgym.SyncVectorEnv([env_fn])
