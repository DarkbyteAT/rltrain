"""Builder for constructing JAX env wrappers from JSON config dicts."""

from __future__ import annotations

from rltrain.env import GymnasiumEnv, GymnaxEnv


def env(
    id: str,
    *,
    backend: str = "gymnax",
    reward_run_rate: float = 0.1,
    num_envs: int = 1,
) -> GymnaxEnv | GymnasiumEnv:
    """Build a wrapped environment.

    Args:
        id: Environment identifier (gymnax env name or gymnasium env id).
        backend: ``"gymnax"`` for pure-JAX, scannable envs; ``"gymnasium"`` for
            the eager Python fallback.
        reward_run_rate: EMA mixing weight for the env-side running return
            (gymnax only).
        num_envs: Vectorisation count. For ``gymnax`` envs, runs ``num_envs``
            parallel copies via ``jax.vmap``. For ``gymnasium`` envs, uses
            ``gymnasium.make_vec(..., num_envs=...)``.

    Returns:
        A ``GymnaxEnv`` or ``GymnasiumEnv`` wrapper.
    """
    if backend == "gymnax":
        return GymnaxEnv(id, reward_run_rate=reward_run_rate, num_envs=num_envs)
    if backend == "gymnasium":
        return GymnasiumEnv(id, num_envs=num_envs)
    raise ValueError(f"unknown backend={backend!r}; expected 'gymnax' or 'gymnasium'")
