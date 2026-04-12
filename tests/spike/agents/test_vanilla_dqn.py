"""Tests for VanillaDQN — pure-functional JAX agent."""

import jax
import jax.numpy as jnp
import optax
import pytest

from spike.agents.vanilla_dqn import DQNState, VanillaDQN
from spike.buffer import buffer_add, buffer_sample, make_buffer
from spike.networks import MLP
from spike.transitions import make_transition


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

OBS_DIM = 4
NUM_ACTIONS = 2
KEY = jax.random.PRNGKey(0)


def _make_agent(key=KEY):
    return VanillaDQN(
        q_net=MLP(OBS_DIM, NUM_ACTIONS, width=64, depth=2, key=key),
        optimizer=optax.adam(1e-3),
        gamma=0.99,
        target_rate=0.01,
        num_actions=NUM_ACTIONS,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=5e-4,
    )


def _make_batch(key, batch_size=32):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return make_transition(
        obs=jax.random.normal(k1, (batch_size, OBS_DIM)),
        action=jax.random.randint(k2, (batch_size,), 0, NUM_ACTIONS),
        reward=jax.random.normal(k3, (batch_size,)),
        next_obs=jax.random.normal(k4, (batch_size, OBS_DIM)),
        done=jnp.zeros(batch_size, dtype=jnp.bool_),
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_q_values_shape():
    """q_values returns one value per action."""
    # Given
    agent = _make_agent()
    state = agent.init(jax.random.PRNGKey(1))
    import equinox as eqx

    model = eqx.combine(state.params, eqx.partition(agent, eqx.is_array)[1])
    obs = jnp.ones(OBS_DIM)

    # When
    q = model.q_net(obs)

    # Then
    assert q.shape == (NUM_ACTIONS,)


@pytest.mark.unit
def test_act_epsilon_greedy():
    """With epsilon=1 actions are uniformly random; with epsilon=0 they are argmax Q."""
    # Given
    agent = _make_agent()
    state = agent.init(jax.random.PRNGKey(1))
    obs = jnp.ones(OBS_DIM)

    # When — fully random
    random_state = DQNState(
        params=state.params,
        opt_state=state.opt_state,
        target_params=state.target_params,
        epsilon=jnp.array(1.0),
    )
    keys = jax.random.split(jax.random.PRNGKey(42), 200)
    random_actions = jnp.array([agent.act(random_state, obs, k) for k in keys])

    # Then — not all the same
    assert jnp.unique(random_actions).shape[0] > 1

    # When — fully greedy
    greedy_state = DQNState(
        params=state.params,
        opt_state=state.opt_state,
        target_params=state.target_params,
        epsilon=jnp.array(0.0),
    )
    greedy_actions = jnp.array([agent.act(greedy_state, obs, k) for k in keys])

    # Then — all identical (argmax is deterministic)
    assert jnp.all(greedy_actions == greedy_actions[0])


@pytest.mark.unit
def test_loss_is_scalar():
    """loss() returns a finite scalar."""
    # Given
    import equinox as eqx

    agent = _make_agent()
    state = agent.init(jax.random.PRNGKey(1))
    batch = _make_batch(jax.random.PRNGKey(1))
    static = eqx.partition(agent, eqx.is_array)[1]

    # When — call _loss on the reconstructed agent
    model = eqx.combine(state.params, static)
    loss_val = model._loss(state.target_params, static, batch)

    # Then
    assert loss_val.shape == ()
    assert jnp.isfinite(loss_val)


@pytest.mark.unit
def test_target_update():
    """After incremental_update, target params differ from both old and new."""
    # Given
    agent = _make_agent(jax.random.PRNGKey(0))
    state = agent.init(jax.random.PRNGKey(1))
    batch = _make_batch(jax.random.PRNGKey(2))

    # When
    new_state, _ = agent.learn(state, batch, jax.random.PRNGKey(0))

    # Then — target differs from both old and new
    for old_t, new_t, new_p in zip(
        jax.tree.leaves(state.target_params),
        jax.tree.leaves(new_state.target_params),
        jax.tree.leaves(new_state.params),
        strict=False,
    ):
        if old_t.size > 0:
            assert not jnp.allclose(old_t, new_t), "target should differ from old"
            assert not jnp.allclose(new_p, new_t), "target should differ from new params"


@pytest.mark.unit
def test_learn_updates_params():
    """One learn step produces different params."""
    # Given
    agent = _make_agent()
    state = agent.init(jax.random.PRNGKey(1))
    batch = _make_batch(jax.random.PRNGKey(2))

    # When
    new_state, metrics = agent.learn(state, batch, jax.random.PRNGKey(0))

    # Then — params changed
    old_flat = jax.tree.leaves(state.params)
    new_flat = jax.tree.leaves(new_state.params)
    any_changed = any(not jnp.allclose(o, n) for o, n in zip(old_flat, new_flat, strict=False))
    assert any_changed, "params should change after one learn step"

    # And — loss is a finite scalar
    assert jnp.isfinite(metrics["loss"])


# ---------------------------------------------------------------------------
# Integration / end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.e2e
def test_trains_cartpole():
    """Train DQN on gymnax CartPole for ~50K steps, achieve return > 50."""

    from spike.env import GymnaxEnv

    # Given — environment, agent, buffer
    env = GymnaxEnv("CartPole-v1")
    key = jax.random.PRNGKey(0)
    k_agent, k_env, key = jax.random.split(key, 3)

    agent = VanillaDQN(
        q_net=MLP(env.obs_shape[0], env.num_actions, width=128, depth=2, key=k_agent),
        optimizer=optax.adam(3e-4),
        gamma=0.99,
        target_rate=0.005,
        num_actions=env.num_actions,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=5e-4,
    )

    state = agent.init(jax.random.PRNGKey(42))
    buffer = make_buffer(capacity=10_000, obs_shape=env.obs_shape, action_shape=())

    # JIT the learn function
    learn_jit = jax.jit(agent.learn)

    # When — training loop
    env_state = env.reset(k_env)
    total_steps = 50_000
    batch_size = 64
    warmup_steps = 500

    for step in range(total_steps):
        key, k_act, k_step, k_sample = jax.random.split(key, 4)

        action = agent.act(state, env_state.obs, k_act)
        prev_obs = env_state.obs
        env_state = env.step(env_state, action, k_step)

        transition = make_transition(
            obs=prev_obs,
            action=action,
            reward=jnp.array(1.0),
            next_obs=env_state.obs,
            done=env_state.done,
        )
        buffer = buffer_add(buffer, transition)

        if step >= warmup_steps and int(buffer.size) >= batch_size:
            batch = buffer_sample(buffer, k_sample, batch_size)
            state, _metrics = learn_jit(state, batch, jax.random.PRNGKey(0))

    # Evaluate: run 20 greedy episodes
    eval_returns: list[float] = []
    for _ep in range(20):
        key, k_reset = jax.random.split(key)
        es = env.reset(k_reset)
        ep_return = 0.0
        greedy_state = DQNState(
            params=state.params,
            opt_state=state.opt_state,
            target_params=state.target_params,
            epsilon=jnp.array(0.0),
        )
        for _t in range(500):
            key, k_act, k_step = jax.random.split(key, 3)
            action = agent.act(greedy_state, es.obs, k_act)
            es = env.step(es, action, k_step)
            ep_return += 1.0
            if es.done:
                break
        eval_returns.append(ep_return)

    mean_return = sum(eval_returns) / len(eval_returns)

    # Then
    assert mean_return > 50, f"Expected mean eval return > 50 after 50K steps, got {mean_return:.1f}"
