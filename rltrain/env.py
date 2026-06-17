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

    def close(self) -> None:
        """Release env resources (gymnasium subprocesses, etc.)."""
        ...


@chex.dataclass
class EnvState:
    """Jittable env state — a pytree of arrays.

    ``running_return`` is an EMA over completed-episode returns, updated only
    when ``done`` fires. Between episodes it carries the previous value so
    downstream metrics see a stable smoothed signal. It is initialised to
    ``NaN`` as a sentinel for "no completed episode yet"; on the first
    ``done`` the EMA is warm-started to the first episode's return rather
    than EMA-blended with zero (which would underweight the first measurement
    by a factor of ``reward_run_rate``). Consumers that read
    ``running_return`` BEFORE the first episode terminates MUST guard with
    ``jnp.isnan(...)`` — the built-in callbacks only consume it at
    ``on_episode_end``, by which point it is always a real number.
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

    Vectorisation: pass ``num_envs > 1`` to run ``num_envs`` parallel copies
    via ``jax.vmap``. ``reset(key)`` and ``step(state, action, key)`` then
    expect / return state with a leading ``num_envs`` axis on every field.
    The single-env path (``num_envs=1``) is unchanged — no leading dim added.
    """

    capabilities = EnvCapabilities(pure_step=True, vmap_batch=True, scan_rollout=True)

    def __init__(self, env_name: str, *, reward_run_rate: float = 0.1, num_envs: int = 1):
        """Initialise from a gymnax environment name (e.g. ``'CartPole-v1'``).

        Args:
            env_name: gymnax environment identifier.
            reward_run_rate: EMA mixing weight for ``running_return`` updates
                on episode completion. Matches the PyTorch MDP's ``run_beta``.
            num_envs: Number of parallel env copies to vmap over. ``1`` keeps
                the single-env semantics (no leading axis).
        """
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self.env, self.env_params = gymnax.make(env_name)
        self.obs_shape: tuple[int, ...] = self.env.obs_shape
        self.num_actions: int = self.env.num_actions
        self.reward_run_rate: float = reward_run_rate
        self.num_envs: int = num_envs

    def reset(self, key: PRNGKeyArray) -> EnvState:
        """Reset the environment. Returns a batched ``EnvState`` when ``num_envs > 1``."""
        if self.num_envs == 1:
            return self._reset_single(key)
        keys = jax.random.split(key, self.num_envs)
        return jax.vmap(self._reset_single)(keys)

    def step(self, state: EnvState, action: chex.Array, key: PRNGKeyArray) -> EnvState:
        """Step the environment, returning a batched ``EnvState`` when ``num_envs > 1``.

        When ``num_envs > 1``, ``state`` and ``action`` must carry a leading
        ``num_envs`` axis; the returned ``EnvState`` does too.
        """
        if self.num_envs == 1:
            return self._step_single(state, action, key)
        keys = jax.random.split(key, self.num_envs)
        return jax.vmap(self._step_single)(state, action, keys)

    def _reset_single(self, key: PRNGKeyArray) -> EnvState:
        """Reset a single env copy and return its initial ``EnvState``."""
        obs, internal = self.env.reset(key, self.env_params)
        return EnvState(
            internal=internal,
            obs=obs,
            done=jnp.array(False),
            reward=jnp.array(0.0),
            episode_return=jnp.array(0.0),
            episode_length=jnp.array(0, dtype=jnp.int32),
            running_return=jnp.array(jnp.nan, dtype=jnp.float32),
        )

    def _step_single(self, state: EnvState, action: chex.Array, key: PRNGKeyArray) -> EnvState:
        """Step a single env copy, auto-resetting on done. Pure: jit/vmap-safe."""
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

        # EMA running_return only updates on episode boundaries. On the very
        # first completed episode the prior ``running_return`` is NaN; warm-
        # start to the episode return so the first measurement isn't blended
        # with zero (which would underweight it by ``reward_run_rate``).
        beta = self.reward_run_rate
        ema_blend = beta * episode_return + (1.0 - beta) * state.running_return
        warm_or_ema = jnp.where(jnp.isnan(state.running_return), episode_return, ema_blend)
        new_running_return = jnp.where(done, warm_or_ema, state.running_return)

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
        """Reset and return initial observation as a JAX array.

        Lazily constructs the underlying gymnasium env on first call and
        reuses it on subsequent resets. Re-instantiating per reset
        leaked the previous env's subprocess pool when ``num_envs > 1``.
        """
        seed = int(jax.random.randint(key, (), 0, 2**30)) if key is not None else None
        if self._env is None:
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
