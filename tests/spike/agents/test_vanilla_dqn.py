"""Tests for VanillaDQN — pure-functional JAX agent."""

import jax
import jax.numpy as jnp
import optax
import pytest

from spike.agents.vanilla_dqn import VanillaDQN, learn
from spike.buffer import buffer_add, buffer_sample, make_buffer
from spike.transitions import make_transition


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

OBS_DIM = 4
NUM_ACTIONS = 2
KEY = jax.random.PRNGKey(0)


def _make_agent(key=KEY):
    return VanillaDQN(
        obs_size=OBS_DIM,
        num_actions=NUM_ACTIONS,
        width=64,
        depth=2,
        gamma=0.99,
        key=key,
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
    obs = jnp.ones(OBS_DIM)

    # When
    q = agent.q_values(obs)

    # Then
    assert q.shape == (NUM_ACTIONS,)


@pytest.mark.unit
def test_act_epsilon_greedy():
    """With epsilon=1 actions are uniformly random; with epsilon=0 they are argmax Q."""
    # Given
    agent = _make_agent()
    obs = jnp.ones(OBS_DIM)

    # When — fully random
    keys = jax.random.split(jax.random.PRNGKey(42), 200)
    random_actions = jnp.array([agent.act(obs, k, epsilon=1.0) for k in keys])

    # Then — not all the same (would be astronomically unlikely for 200 draws)
    assert jnp.unique(random_actions).shape[0] > 1

    # When — fully greedy
    greedy_actions = jnp.array([agent.act(obs, k, epsilon=0.0) for k in keys])

    # Then — all identical (argmax is deterministic)
    assert jnp.all(greedy_actions == greedy_actions[0])

    # And — the greedy action matches argmax of Q
    q = agent.q_values(obs)
    assert greedy_actions[0] == jnp.argmax(q)


@pytest.mark.unit
def test_loss_is_scalar():
    """loss() returns a finite scalar."""
    # Given
    agent = _make_agent()
    import equinox as eqx

    target_params, _ = eqx.partition(agent, eqx.is_array)
    batch = _make_batch(jax.random.PRNGKey(1))

    # When
    loss_val = agent.loss(target_params, batch)

    # Then
    assert loss_val.shape == ()
    assert jnp.isfinite(loss_val)


@pytest.mark.unit
def test_target_update():
    """After incremental_update, target params differ from both old and new."""
    # Given
    import equinox as eqx

    agent = _make_agent(jax.random.PRNGKey(0))
    agent2 = _make_agent(jax.random.PRNGKey(99))
    old_target, _ = eqx.partition(agent, eqx.is_array)
    new_params, _ = eqx.partition(agent2, eqx.is_array)

    # When
    updated = optax.incremental_update(new_params, old_target, step_size=0.1)

    # Then — updated differs from both old and new
    old_flat = jax.tree.leaves(old_target)
    new_flat = jax.tree.leaves(new_params)
    upd_flat = jax.tree.leaves(updated)

    for o, n, u in zip(old_flat, new_flat, upd_flat, strict=False):
        assert not jnp.allclose(o, u), "updated should differ from old target"
        assert not jnp.allclose(n, u), "updated should differ from new params"


@pytest.mark.unit
def test_learn_updates_params():
    """One learn step produces different params."""
    # Given
    import equinox as eqx

    agent = _make_agent()
    params, static = eqx.partition(agent, eqx.is_array)
    target_params, _ = eqx.partition(agent, eqx.is_array)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    batch = _make_batch(jax.random.PRNGKey(2))

    # When
    new_params, new_opt_state, new_target, metrics = learn(
        params, static, target_params, opt_state, optimizer, batch, target_rate=0.01
    )

    # Then — params changed
    old_flat = jax.tree.leaves(params)
    new_flat = jax.tree.leaves(new_params)
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
    import equinox as eqx

    from spike.env import GymnaxEnv

    # Given — environment, agent, buffer, optimizer
    env = GymnaxEnv("CartPole-v1")
    key = jax.random.PRNGKey(0)
    k_agent, k_env, key = jax.random.split(key, 3)

    agent = VanillaDQN(
        obs_size=env.obs_shape[0],
        num_actions=env.num_actions,
        width=128,
        depth=2,
        gamma=0.99,
        key=k_agent,
    )

    params, static = eqx.partition(agent, eqx.is_array)
    target_params, _ = eqx.partition(agent, eqx.is_array)
    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(params)
    buffer = make_buffer(capacity=10_000, obs_shape=env.obs_shape, action_shape=())

    # Epsilon schedule: linear decay from 1.0 to 0.05 over 40K steps
    eps_start, eps_end, eps_decay_steps = 1.0, 0.05, 40_000

    # JIT the learn function
    learn_jit = jax.jit(learn, static_argnames=("static", "optimizer", "target_rate"))

    # When — training loop
    env_state = env.reset(k_env)
    total_steps = 50_000
    batch_size = 64
    warmup_steps = 500

    for step in range(total_steps):
        key, k_act, k_step, k_sample = jax.random.split(key, 4)

        # Epsilon schedule
        epsilon = max(eps_end, eps_start - (eps_start - eps_end) * step / eps_decay_steps)

        # Act (agent always reflects current params)
        action = agent.act(env_state.obs, k_act, epsilon)

        # Step environment
        prev_obs = env_state.obs
        env_state = env.step(env_state, action, k_step)

        # Store transition (CartPole reward = +1 per step)
        transition = make_transition(
            obs=prev_obs,
            action=action,
            reward=jnp.array(1.0),
            next_obs=env_state.obs,
            done=env_state.done,
        )
        buffer = buffer_add(buffer, transition)

        # Learn after warmup
        if step >= warmup_steps and int(buffer.size) >= batch_size:
            batch = buffer_sample(buffer, k_sample, batch_size)

            params, opt_state, target_params, metrics = learn_jit(
                params,
                static=static,
                target_params=target_params,
                opt_state=opt_state,
                optimizer=optimizer,
                batch=batch,
                target_rate=0.005,
            )
            agent = eqx.combine(params, static)

    # Evaluate: run 20 greedy episodes
    eval_returns: list[float] = []
    for ep in range(20):
        key, k_reset = jax.random.split(key)
        es = env.reset(k_reset)
        ep_return = 0.0
        for t in range(500):
            key, k_act, k_step = jax.random.split(key, 3)
            action = agent.act(es.obs, k_act, epsilon=0.0)
            es = env.step(es, action, k_step)
            ep_return += 1.0
            if es.done:
                break
        eval_returns.append(ep_return)

    mean_return = sum(eval_returns) / len(eval_returns)

    # Then
    assert mean_return > 50, f"Expected mean eval return > 50 after 50K steps, got {mean_return:.1f}"
