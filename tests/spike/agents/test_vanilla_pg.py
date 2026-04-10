"""Tests for the pure-functional VanillaPG agent."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from spike.agents.vanilla_pg import VanillaPG, discount, learn
from spike.env import GymnaxEnv
from spike.heads import DiscreteHead
from spike.networks import MLP
from spike.transitions import Transition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OBS_DIM = 4
NUM_ACTIONS = 2
HIDDEN = 32


def _make_agent(key: jax.Array) -> VanillaPG:
    """Build a small VanillaPG for CartPole-sized problems."""
    k1, k2 = jax.random.split(key)
    return VanillaPG(
        actor=MLP(OBS_DIM, HIDDEN, width=HIDDEN, depth=1, key=k1),
        action_head=DiscreteHead(HIDDEN, NUM_ACTIONS, key=k2),
        gamma=0.99,
        tau=0.01,
        normalise=True,
    )


def _make_transitions(key: jax.Array, n: int = 16) -> Transition:
    """Fabricate a batch of random transitions."""
    k1, k2, k3 = jax.random.split(key, 3)
    return Transition(
        obs=jax.random.normal(k1, (n, OBS_DIM)),
        action=jax.random.randint(k2, (n, 1), 0, NUM_ACTIONS),
        reward=jax.random.normal(k3, (n,)),
        next_obs=jax.random.normal(k1, (n, OBS_DIM)),
        done=jnp.zeros(n, dtype=jnp.bool_),
        log_prob=jnp.zeros(n),
        value=jnp.zeros(n),
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_discount_basic():
    """Given constant rewards with no episode boundaries, discount produces
    the geometric series."""
    # Given
    rewards = jnp.ones(5)
    dones = jnp.zeros(5)
    gamma = 0.5

    # When
    returns = discount(rewards, dones, gamma)

    # Then — G_0 = 1 + 0.5 + 0.25 + 0.125 + 0.0625 = 1.9375
    assert jnp.allclose(returns[0], 1.9375, atol=1e-5)
    assert jnp.allclose(returns[-1], 1.0, atol=1e-5)


def test_discount_resets_at_done():
    """Given a done flag mid-sequence, discount resets the accumulator."""
    # Given
    rewards = jnp.array([1.0, 1.0, 1.0, 1.0])
    dones = jnp.array([0.0, 1.0, 0.0, 0.0])
    gamma = 0.99

    # When
    returns = discount(rewards, dones, gamma)

    # Then — the return at index 0 should NOT see rewards past the done at index 1
    # G_0 = r_0 + gamma * 0 (because done[1]=1 resets) = ... actually:
    # scan reverse: t=3: acc=1.0; t=2: acc=1+0.99*1=1.99; t=1: acc=1+0.99*1.99*0=1.0 (done resets); t=0: acc=1+0.99*1=1.99
    assert jnp.allclose(returns[0], 1.99, atol=1e-5)
    assert jnp.allclose(returns[1], 1.0, atol=1e-5)
    assert jnp.allclose(returns[2], 1.0 + 0.99 * 1.0, atol=1e-5)


def test_act_returns_valid_action():
    """Given an observation, act produces an action in valid range and a scalar log_prob."""
    # Given
    key = jax.random.PRNGKey(0)
    agent = _make_agent(key)
    obs = jnp.ones(OBS_DIM)

    # When
    action, log_prob, dist = agent.act(obs, jax.random.PRNGKey(1))

    # Then
    assert action.shape == ()
    assert 0 <= int(action) < NUM_ACTIONS
    assert log_prob.shape == ()
    assert jnp.isfinite(log_prob)


def test_loss_is_scalar():
    """Given a batch of transitions, loss returns a finite scalar."""
    # Given
    key = jax.random.PRNGKey(42)
    agent = _make_agent(key)
    transitions = _make_transitions(jax.random.PRNGKey(1))

    # When
    loss_val = agent.loss(transitions)

    # Then
    assert loss_val.shape == ()
    assert jnp.isfinite(loss_val)


def test_gradients_flow():
    """Gradients through loss are non-zero — the loss depends on parameters."""
    # Given
    key = jax.random.PRNGKey(7)
    agent = _make_agent(key)
    transitions = _make_transitions(jax.random.PRNGKey(2))

    # When
    _loss, grads = eqx.filter_value_and_grad(lambda m: m.loss(transitions))(agent)

    # Then — at least some gradient leaves are non-zero
    grad_leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
    has_nonzero = any(jnp.any(g != 0.0) for g in grad_leaves)
    assert has_nonzero, "All gradients are zero — loss is disconnected from parameters"


def test_learn_updates_params():
    """After one learn step, at least some parameters differ from the originals."""
    # Given
    key = jax.random.PRNGKey(99)
    agent = _make_agent(key)
    optimizer = optax.adam(1e-3)
    params, static = eqx.partition(agent, eqx.is_array)
    opt_state = optimizer.init(params)
    transitions = _make_transitions(jax.random.PRNGKey(3))

    # When
    new_params, _new_opt_state, metrics = learn(params, static, opt_state, optimizer, transitions)

    # Then
    assert jnp.isfinite(metrics["loss"])
    old_leaves = jax.tree.leaves(params)
    new_leaves = jax.tree.leaves(new_params)
    any_changed = any(not jnp.allclose(o, n) for o, n in zip(old_leaves, new_leaves, strict=False))
    assert any_changed, "No parameters changed after a learn step"


# ---------------------------------------------------------------------------
# End-to-end training test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_trains_cartpole():
    """Train VanillaPG on CartPole-v1 (gymnax) for ~50K steps.

    Success criterion: mean return over the last 20 episodes exceeds 50.
    This is not perfect play (max 500) — just evidence of a learning signal.
    """
    # Given
    env = GymnaxEnv("CartPole-v1")
    key = jax.random.PRNGKey(0)
    k_agent, k_env, k_loop = jax.random.split(key, 3)

    agent = VanillaPG(
        actor=MLP(env.obs_shape[0], 64, width=64, depth=1, key=jax.random.split(k_agent)[0]),
        action_head=DiscreteHead(64, env.num_actions, key=jax.random.split(k_agent)[1]),
        gamma=0.99,
        tau=0.01,
        normalise=True,
    )

    optimizer = optax.adam(3e-3)
    params, static = eqx.partition(agent, eqx.is_array)
    opt_state = optimizer.init(params)

    env_state = env.reset(k_env)

    episode_returns: list[float] = []
    total_steps = 0
    max_steps = 50_000

    # When — Python collection loop, jitted learn
    jit_learn = jax.jit(lambda p, s, os, t: learn(p, s, os, optimizer, t))

    while total_steps < max_steps:
        # Collect one episode into a Python list, then stack
        k_loop, k_ep = jax.random.split(k_loop)
        ep_obs, ep_act, ep_rew, ep_next, ep_done, ep_lp = [], [], [], [], [], []
        ep_return = 0.0

        while True:
            k_ep, k_act, k_step = jax.random.split(k_ep, 3)

            current_agent = eqx.combine(params, static)
            action, log_prob, _dist = current_agent.act(env_state.obs, k_act)

            prev_obs = env_state.obs
            env_state = env.step(env_state, action, k_step)

            ep_obs.append(prev_obs)
            ep_act.append(action.reshape(1))
            ep_rew.append(env_state.reward)
            ep_next.append(env_state.obs)
            ep_done.append(env_state.done)
            ep_lp.append(log_prob)
            ep_return += float(env_state.reward)
            total_steps += 1

            if bool(env_state.done) or len(ep_obs) >= 500:
                break

        episode_returns.append(ep_return)

        # Stack into a single Transition batch
        transitions = Transition(
            obs=jnp.stack(ep_obs),
            action=jnp.stack(ep_act),
            reward=jnp.stack(ep_rew),
            next_obs=jnp.stack(ep_next),
            done=jnp.stack(ep_done),
            log_prob=jnp.stack(ep_lp),
            value=jnp.zeros(len(ep_obs)),
        )
        params, opt_state, _metrics = jit_learn(params, static, opt_state, transitions)

    # Then
    recent = episode_returns[-20:]
    mean_return = sum(recent) / len(recent)
    assert mean_return > 50, (
        f"Mean return over last 20 episodes was {mean_return:.1f}, expected > 50. "
        f"Full return history (last 40): {episode_returns[-40:]}"
    )
