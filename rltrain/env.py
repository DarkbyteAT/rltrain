"""Environment wrappers — gymnasium fallback and gymnax jittable envs.

Two wrappers behind a common capability protocol. The trainer dispatches
on ``env.capabilities`` to select the right rollout strategy.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Protocol, runtime_checkable

import chex
import gymnasium
import gymnax
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray


class EnvCapabilities(NamedTuple):
    """What JAX transformations the env supports."""

    pure_step: bool
    vmap_batch: bool
    scan_rollout: bool


@runtime_checkable
class Env(Protocol):
    """Structural contract every environment wrapper satisfies.

    The trainer dispatches on ``capabilities`` to pick the right loop
    strategy. ``reset`` and ``step`` differ between the gymnax (pure
    JAX, state-carrying) and gymnasium (eager Python) backends — the
    Protocol pins them as ``Callable[..., Any]`` so concrete callers
    can pass whatever the chosen backend expects.
    """

    capabilities: EnvCapabilities

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        """Reset the environment. Signature varies by backend."""
        ...

    def step(self, *args: Any, **kwargs: Any) -> Any:
        """Step the environment. Signature varies by backend."""
        ...


@chex.dataclass
class EnvState:
    """Jittable env state — a pytree of arrays.

    ``running_return`` is an EMA over completed-episode returns, updated only
    when ``done`` fires. Between episodes it carries the previous value so
    downstream metrics see a stable smoothed signal.
    """

    internal: Any  # gymnax env-specific state
    obs: chex.Array
    done: chex.Array
    reward: chex.Array
    episode_return: chex.Array
    episode_length: chex.Array
    running_return: chex.Array


class GymnaxEnv:
    """Wrapper around a gymnax environment exposing a pure-function step.

    All methods are pure functions suitable for ``jit``, ``vmap``, and ``scan``.
    """

    capabilities = EnvCapabilities(pure_step=True, vmap_batch=True, scan_rollout=True)

    def __init__(self, env_name: str, *, reward_run_rate: float = 0.1):
        """Initialise from a gymnax environment name (e.g. ``'CartPole-v1'``).

        Args:
            env_name: gymnax environment identifier.
            reward_run_rate: EMA mixing weight for ``running_return`` updates
                on episode completion. Matches the PyTorch MDP's ``run_beta``.
        """
        self.env, self.env_params = gymnax.make(env_name)
        self.obs_shape: tuple[int, ...] = self.env.obs_shape
        self.num_actions: int = self.env.num_actions
        self.reward_run_rate: float = reward_run_rate

    def reset(self, key: PRNGKeyArray) -> EnvState:
        """Reset the environment, returning initial state."""
        obs, internal = self.env.reset(key, self.env_params)
        return EnvState(
            internal=internal,
            obs=obs,
            done=jnp.array(False),
            reward=jnp.array(0.0),
            episode_return=jnp.array(0.0),
            episode_length=jnp.array(0, dtype=jnp.int32),
            running_return=jnp.array(0.0),
        )

    def step(self, state: EnvState, action: chex.Array, key: PRNGKeyArray) -> EnvState:
        """Take one step, auto-resetting on done."""
        obs, internal, reward, done, info = self.env.step(key, state.internal, action, self.env_params)
        episode_return = state.episode_return + reward
        episode_length = state.episode_length + 1

        # Auto-reset: if done, reset the env but keep the terminal metrics
        reset_key = jax.random.fold_in(key, 1)
        reset_obs, reset_internal = self.env.reset(reset_key, self.env_params)

        new_internal = jax.tree.map(lambda r, c: jnp.where(done, r, c), reset_internal, internal)
        new_obs = jnp.where(done, reset_obs, obs)
        new_episode_return = jnp.where(done, 0.0, episode_return)
        new_episode_length = jnp.where(done, 0, episode_length)

        # EMA running_return only updates on episode boundaries
        beta = self.reward_run_rate
        ema_new = beta * episode_return + (1.0 - beta) * state.running_return
        new_running_return = jnp.where(done, ema_new, state.running_return)

        return EnvState(
            internal=new_internal,
            obs=new_obs,
            done=done,
            reward=reward,
            episode_return=new_episode_return,
            episode_length=new_episode_length,
            running_return=new_running_return,
        )


class GymnasiumEnv:
    """Wrapper around a gymnasium environment for the Python-loop fallback.

    Not jittable. The step function converts between numpy and JAX arrays
    at the boundary.
    """

    capabilities = EnvCapabilities(pure_step=False, vmap_batch=False, scan_rollout=False)

    def __init__(self, env_id: str, num_envs: int = 1):
        """Initialise with a gymnasium environment ID."""
        self.env_id = env_id
        self.num_envs = num_envs
        self._env: gymnasium.Env | None = None

    @property
    def obs_shape(self) -> tuple[int, ...]:
        """Observation shape (single env)."""
        env = gymnasium.make(self.env_id)
        shape = env.observation_space.shape
        env.close()
        assert shape is not None
        return shape

    @property
    def num_actions(self) -> int:
        """Number of discrete actions."""
        env = gymnasium.make(self.env_id)
        n = env.action_space.n  # type: ignore[attr-defined]
        env.close()
        return int(n)

    def reset(self, key: PRNGKeyArray | None = None) -> chex.Array:
        """Reset and return initial observation as a JAX array."""
        seed = int(jax.random.randint(key, (), 0, 2**30)) if key is not None else None
        if self.num_envs > 1:
            self._env = gymnasium.make_vec(self.env_id, num_envs=self.num_envs)
        else:
            self._env = gymnasium.make(self.env_id)
        obs, _ = self._env.reset(seed=seed)
        return jnp.asarray(obs, dtype=jnp.float32)

    def step(self, action: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array, dict[str, Any]]:
        """Step the env with a numpy action, return JAX arrays."""
        assert self._env is not None
        import numpy as np

        action_np = np.asarray(action)
        obs, reward, terminated, truncated, info = self._env.step(action_np)
        done = terminated | truncated

        return (
            jnp.asarray(obs, dtype=jnp.float32),
            jnp.asarray(reward, dtype=jnp.float32),
            jnp.asarray(done, dtype=jnp.bool_),
            info,
        )

    def close(self) -> None:
        """Clean up the underlying gymnasium env."""
        if self._env is not None:
            self._env.close()
            self._env = None
