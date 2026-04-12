"""Tests for the Vanilla Actor-Critic agent with TD error advantage."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from spike.agents.agent import Agent
from spike.agents.vanilla_ac import VanillaAC
from spike.heads import DiscreteHead
from spike.networks import MLP
from spike.transitions import Transition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OBS_DIM = 4
NUM_ACTIONS = 2
HIDDEN = 32


def _make_agent(key: jax.Array) -> VanillaAC:
    """Build a small VanillaAC agent for CartPole-sized problems."""
    k1, k2, k3 = jax.random.split(key, 3)
    return VanillaAC(
        actor=MLP(OBS_DIM, HIDDEN, width=HIDDEN, depth=1, key=k1),
        action_head=DiscreteHead(HIDDEN, NUM_ACTIONS, key=k2),
        critic=MLP(OBS_DIM, 1, width=HIDDEN, depth=1, key=k3),
        optimizer=optax.adam(1e-3),
        gamma=0.99,
        tau=0.01,
        beta_critic=0.5,
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


@pytest.mark.unit
def test_loss_is_scalar():
    """Given a batch of transitions, loss returns a finite scalar."""
    # Given
    agent = _make_agent(jax.random.PRNGKey(42))
    transitions = _make_transitions(jax.random.PRNGKey(1))

    # When
    loss_val = agent._loss(transitions)

    # Then
    assert loss_val.shape == ()
    assert jnp.isfinite(loss_val)


@pytest.mark.unit
def test_gradients_flow():
    """Gradients through loss are non-zero."""
    # Given
    agent = _make_agent(jax.random.PRNGKey(7))
    transitions = _make_transitions(jax.random.PRNGKey(2))

    # When
    _loss, grads = eqx.filter_value_and_grad(lambda m: m._loss(transitions))(agent)

    # Then
    grad_leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
    has_nonzero = any(jnp.any(g != 0.0) for g in grad_leaves)
    assert has_nonzero, "All gradients are zero"


@pytest.mark.unit
def test_learn_updates_params():
    """After one learn step, at least some parameters differ."""
    # Given
    agent = _make_agent(jax.random.PRNGKey(99))
    state = agent.init(jax.random.PRNGKey(1))
    transitions = _make_transitions(jax.random.PRNGKey(3))

    # When
    new_state, metrics = agent.learn(state, transitions)

    # Then
    assert jnp.isfinite(metrics["loss"])
    old_leaves = jax.tree.leaves(state.params)
    new_leaves = jax.tree.leaves(new_state.params)
    any_changed = any(not jnp.allclose(o, n) for o, n in zip(old_leaves, new_leaves, strict=False))
    assert any_changed, "No parameters changed after a learn step"


@pytest.mark.unit
def test_act_returns_valid_action():
    """Given an observation, act produces an action in valid range."""
    # Given
    agent = _make_agent(jax.random.PRNGKey(0))
    state = agent.init(jax.random.PRNGKey(1))
    obs = jnp.ones(OBS_DIM)

    # When
    action = agent.act(state, obs, jax.random.PRNGKey(2))

    # Then
    assert action.shape == ()
    assert 0 <= int(action) < NUM_ACTIONS


@pytest.mark.unit
def test_satisfies_agent_protocol():
    """VanillaAC satisfies the Agent protocol via structural subtyping."""
    # Given
    agent = _make_agent(jax.random.PRNGKey(0))

    # Then
    assert isinstance(agent, Agent)
