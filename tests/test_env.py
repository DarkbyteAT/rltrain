"""Tests for environment wrappers."""

import jax
import jax.numpy as jnp
import pytest

from rltrain.env import GymnasiumEnv, GymnaxEnv


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


def test_gymnax_env_reset(key):
    """Given a gymnax env, reset returns a valid EnvState."""
    env = GymnaxEnv("CartPole-v1")
    state = env.reset(key)

    assert state.obs.shape == (4,)
    assert state.done == False  # noqa: E712
    assert state.episode_return == 0.0
    assert state.episode_length == 0


def test_gymnax_env_step(key):
    """Given a gymnax env state, step returns a new state with updated fields."""
    env = GymnaxEnv("CartPole-v1")
    state = env.reset(key)
    step_key = jax.random.PRNGKey(1)

    new_state = env.step(state, jnp.array(0), step_key)
    assert new_state.obs.shape == (4,)
    assert new_state.episode_length == 1


def test_gymnax_env_jittable(key):
    """The gymnax env step compiles under jit."""
    env = GymnaxEnv("CartPole-v1")
    state = env.reset(key)

    @jax.jit
    def do_step(state, key):
        return env.step(state, jnp.array(1), key)

    new_state = do_step(state, key)
    assert new_state.obs.shape == (4,)


def test_gymnax_env_vmappable(key):
    """The gymnax env step composes with vmap over multiple envs."""
    env = GymnaxEnv("CartPole-v1")
    num_envs = 4
    keys = jax.random.split(key, num_envs)
    states = jax.vmap(env.reset)(keys)

    assert states.obs.shape == (num_envs, 4)

    step_keys = jax.random.split(jax.random.PRNGKey(1), num_envs)
    actions = jnp.zeros(num_envs, dtype=jnp.int32)
    new_states = jax.vmap(env.step)(states, actions, step_keys)

    assert new_states.obs.shape == (num_envs, 4)


def test_gymnasium_env_reset(key):
    """Given a gymnasium env, reset returns a JAX array observation."""
    env = GymnasiumEnv("CartPole-v1")
    obs = env.reset(key)

    assert obs.shape == (4,)
    assert obs.dtype == jnp.float32
    env.close()


def test_gymnasium_env_step(key):
    """Given a gymnasium env, step returns JAX arrays."""
    env = GymnasiumEnv("CartPole-v1")
    env.reset(key)
    action = jnp.array(0)

    next_obs, reward, done, info = env.step(action)
    assert next_obs.shape == (4,)
    assert reward.dtype == jnp.float32
    assert done.dtype == jnp.bool_
    env.close()


def test_gymnax_env_capabilities():
    """GymnaxEnv reports full capabilities."""
    env = GymnaxEnv("CartPole-v1")
    assert env.capabilities.pure_step is True
    assert env.capabilities.vmap_batch is True
    assert env.capabilities.scan_rollout is True


def test_gymnasium_env_capabilities():
    """GymnasiumEnv reports no capabilities."""
    env = GymnasiumEnv("CartPole-v1")
    assert env.capabilities.pure_step is False
    assert env.capabilities.vmap_batch is False
    assert env.capabilities.scan_rollout is False
