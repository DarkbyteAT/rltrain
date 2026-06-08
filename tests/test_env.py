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
    # running_return is NaN-initialised so the first episode boundary can
    # warm-start the EMA rather than blending the first measurement with zero.
    assert jnp.isnan(state.running_return)


@pytest.mark.unit
def test_gymnax_env_running_return_warm_starts_on_first_episode(key):
    """Given a fresh GymnaxEnv, the first completed episode's return MUST flow
    through to ``running_return`` directly — not be EMA-blended with the
    initial zero (which would underweight it by ``reward_run_rate``)."""
    # Given a CartPole env (random actions end episodes in <500 steps).
    env = GymnaxEnv("CartPole-v1", reward_run_rate=0.1)
    state = env.reset(key)
    rollout_key = jax.random.PRNGKey(123)

    # When we step until the first done fires.
    first_return = None
    for _ in range(500):
        rollout_key, k_act, k_step = jax.random.split(rollout_key, 3)
        action = jax.random.randint(k_act, (), 0, env.num_actions)
        new_state = env.step(state, action, k_step)
        if bool(new_state.done):
            # Pre-reset accumulated return is in state.episode_return + reward.
            first_return = float(state.episode_return) + float(new_state.reward)
            state = new_state
            break
        state = new_state
    assert first_return is not None, "first episode did not terminate within 500 steps"

    # Then running_return must equal the first episode's return exactly — NOT
    # the cold-start `beta * first_return + (1-beta) * 0 = 0.1 * first_return`.
    assert float(state.running_return) == pytest.approx(first_return, rel=1e-5)


@pytest.mark.unit
def test_gymnax_env_running_return_emas_after_warm_start(key):
    """After the warm-start, running_return must EMA-blend subsequent episodes
    against the prior running_return per ``reward_run_rate``."""
    # Given a GymnaxEnv with reward_run_rate=0.3 (chosen distinct from defaults
    # so an off-by-one in the formula would surface).
    beta = 0.3
    env = GymnaxEnv("CartPole-v1", reward_run_rate=beta)
    state = env.reset(key)
    rollout_key = jax.random.PRNGKey(7)

    # When we run for two completed episodes.
    returns = []
    pre_step_return = 0.0
    for _ in range(2000):
        rollout_key, k_act, k_step = jax.random.split(rollout_key, 3)
        action = jax.random.randint(k_act, (), 0, env.num_actions)
        new_state = env.step(state, action, k_step)
        if bool(new_state.done):
            ep_ret = pre_step_return + float(new_state.reward)
            returns.append((ep_ret, float(new_state.running_return)))
            pre_step_return = 0.0
        else:
            pre_step_return = float(new_state.episode_return)
        state = new_state
        if len(returns) >= 2:
            break
    assert len(returns) >= 2, "did not observe two episodes within 2000 steps"

    # Then: episode 0's running_return == episode 0's return (warm-start);
    # episode 1's running_return == beta * ep1_return + (1-beta) * ep0_return.
    ep0_ret, ep0_running = returns[0]
    ep1_ret, ep1_running = returns[1]
    assert ep0_running == pytest.approx(ep0_ret, rel=1e-5)
    expected_ep1_running = beta * ep1_ret + (1.0 - beta) * ep0_ret
    assert ep1_running == pytest.approx(expected_ep1_running, rel=1e-5)


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
