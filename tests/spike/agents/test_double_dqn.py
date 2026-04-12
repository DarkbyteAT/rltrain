"""Tests for DoubleDQN — decoupled action selection and evaluation."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from spike.agents.agent import Agent
from spike.agents.double_dqn import DoubleDQN
from spike.agents.vanilla_dqn import VanillaDQN
from spike.networks import MLP
from spike.transitions import make_transition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OBS_DIM = 4
NUM_ACTIONS = 2
KEY = jax.random.PRNGKey(0)


def _make_agent(key=KEY):
    return DoubleDQN(
        q_net=MLP(OBS_DIM, NUM_ACTIONS, width=64, depth=2, key=key),
        optimizer=optax.adam(1e-3),
        gamma=0.99,
        target_rate=0.01,
        num_actions=NUM_ACTIONS,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=5e-4,
    )


def _make_vanilla(key=KEY):
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
def test_loss_is_scalar():
    """_loss returns a finite scalar."""
    # Given
    agent = _make_agent()
    state = agent.init(jax.random.PRNGKey(1))
    batch = _make_batch(jax.random.PRNGKey(1))
    static = eqx.partition(agent, eqx.is_array)[1]

    # When
    model = eqx.combine(state.params, static)
    loss_val = model._loss(state.target_params, static, batch)

    # Then
    assert loss_val.shape == ()
    assert jnp.isfinite(loss_val)


@pytest.mark.unit
def test_gradients_flow():
    """Non-zero gradients exist through the loss."""
    # Given
    agent = _make_agent()
    state = agent.init(jax.random.PRNGKey(1))
    batch = _make_batch(jax.random.PRNGKey(2))
    static = eqx.partition(agent, eqx.is_array)[1]

    # When
    def loss_fn(params):
        model = eqx.combine(params, static)
        return model._loss(state.target_params, static, batch)

    grads = jax.grad(loss_fn)(state.params)

    # Then
    grad_leaves = jax.tree.leaves(grads)
    has_nonzero = any(jnp.any(g != 0.0) for g in grad_leaves)
    assert has_nonzero, "All gradients are zero"


@pytest.mark.unit
def test_learn_updates_params():
    """Params change after one learn step."""
    # Given
    agent = _make_agent()
    state = agent.init(jax.random.PRNGKey(1))
    batch = _make_batch(jax.random.PRNGKey(2))

    # When
    new_state, metrics = agent.learn(state, batch)

    # Then
    old_flat = jax.tree.leaves(state.params)
    new_flat = jax.tree.leaves(new_state.params)
    any_changed = any(not jnp.allclose(o, n) for o, n in zip(old_flat, new_flat, strict=False))
    assert any_changed, "Params should change after one learn step"
    assert jnp.isfinite(metrics["loss"])


@pytest.mark.unit
def test_act_returns_valid_action():
    """Action is in the valid range [0, num_actions)."""
    # Given
    agent = _make_agent()
    state = agent.init(jax.random.PRNGKey(1))
    obs = jnp.ones(OBS_DIM)

    # When
    action = agent.act(state, obs, jax.random.PRNGKey(2))

    # Then
    assert action.shape == ()
    assert 0 <= int(action) < NUM_ACTIONS


@pytest.mark.unit
def test_satisfies_agent_protocol():
    """DoubleDQN satisfies the Agent protocol."""
    # Given
    agent = _make_agent()

    # Then
    assert isinstance(agent, Agent), "DoubleDQN should satisfy Agent protocol"


@pytest.mark.unit
def test_double_q_differs_from_vanilla():
    """DoubleDQN's target computation gives different values than VanillaDQN's.

    Both agents start with identical weights, then we diverge online from
    target via several learn steps.  With diverged params, the double-Q
    trick (online selects, target evaluates) produces a different loss
    than vanilla (target selects and evaluates).
    """
    # Given — same key so same network weights
    key = jax.random.PRNGKey(42)
    double_agent = _make_agent(key)
    vanilla_agent = _make_vanilla(key)

    # Initialise both with the same state
    double_state = double_agent.init(jax.random.PRNGKey(1))
    vanilla_state = vanilla_agent.init(jax.random.PRNGKey(1))

    # Run several learn steps to diverge online from target
    for i in range(6):
        batch = _make_batch(jax.random.PRNGKey(10 + i))
        double_state, _ = double_agent.learn(double_state, batch)
        vanilla_state, _ = vanilla_agent.learn(vanilla_state, batch)

    # When — compute losses on a fresh batch with the diverged states
    eval_batch = _make_batch(jax.random.PRNGKey(99))

    double_static = eqx.partition(double_agent, eqx.is_array)[1]
    vanilla_static = eqx.partition(vanilla_agent, eqx.is_array)[1]

    double_model = eqx.combine(double_state.params, double_static)
    vanilla_model = eqx.combine(vanilla_state.params, vanilla_static)

    double_loss = double_model._loss(double_state.target_params, double_static, eval_batch)
    vanilla_loss = vanilla_model._loss(vanilla_state.target_params, vanilla_static, eval_batch)

    # Then — losses differ because target computation differs
    assert not jnp.allclose(double_loss, vanilla_loss, atol=1e-6), (
        f"DoubleDQN loss ({float(double_loss):.6f}) should differ from VanillaDQN loss ({float(vanilla_loss):.6f})"
    )
