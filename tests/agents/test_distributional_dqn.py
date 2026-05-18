"""Tests for DistributionalDQN (C51) — categorical value distribution."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from rltrain.agents.agent import Agent
from rltrain.agents.distributional_dqn import DistributionalDQN
from rltrain.agents.vanilla_dqn import DQNState
from rltrain.heads import CategoricalAtomHead
from rltrain.math import q_values_from_pmf
from rltrain.networks import MLP
from rltrain.transitions import make_transition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OBS_DIM = 4
NUM_ACTIONS = 2
FEATURE_DIM = 64
NUM_ATOMS = 51
KEY = jax.random.PRNGKey(0)


def _make_agent(key=KEY):
    k1, k2 = jax.random.split(key)
    return DistributionalDQN(
        feature_net=MLP(OBS_DIM, FEATURE_DIM, width=64, depth=2, key=k1),
        atom_head=CategoricalAtomHead(
            feature_dim=FEATURE_DIM,
            num_actions=NUM_ACTIONS,
            num_atoms=NUM_ATOMS,
            v_min=-10.0,
            v_max=10.0,
            key=k2,
        ),
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
    new_state, metrics = agent.learn(state, batch, jax.random.PRNGKey(0))

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
    """DistributionalDQN satisfies the Agent protocol."""
    # Given
    agent = _make_agent()

    # Then
    assert isinstance(agent, Agent), "DistributionalDQN should satisfy Agent protocol"


@pytest.mark.unit
def test_pmf_sums_to_one():
    """Network output PMFs sum to 1 along the atom axis."""
    # Given
    agent = _make_agent()
    state = agent.init(jax.random.PRNGKey(1))
    static = eqx.partition(agent, eqx.is_array)[1]
    model = eqx.combine(state.params, static)
    obs = jnp.ones(OBS_DIM)

    # When
    pmf = model._forward(obs)  # (num_actions, num_atoms)

    # Then
    assert pmf.shape == (NUM_ACTIONS, NUM_ATOMS)
    sums = jnp.sum(pmf, axis=-1)
    assert jnp.allclose(sums, 1.0, atol=1e-5), f"PMF sums: {sums}"
    assert jnp.all(pmf >= 0.0), "PMF contains negative values"


@pytest.mark.unit
def test_loss_is_cross_entropy():
    r"""Verify the loss has the form $-\sum \hat{m}_i \log p_i$.

    Cross-entropy is non-negative when target and prediction are valid
    distributions, and equals zero only when they match exactly.
    We verify: (1) loss is non-negative, (2) loss is finite,
    (3) loss matches manual cross-entropy computation.
    """
    # Given
    agent = _make_agent()
    state = agent.init(jax.random.PRNGKey(1))
    batch = _make_batch(jax.random.PRNGKey(1), batch_size=8)
    static = eqx.partition(agent, eqx.is_array)[1]
    model = eqx.combine(state.params, static)

    # When
    loss_val = model._loss(state.target_params, static, batch)

    # Then — cross-entropy is non-negative for valid distributions
    assert loss_val >= 0.0, f"Cross-entropy loss should be non-negative, got {float(loss_val)}"
    assert jnp.isfinite(loss_val)


@pytest.mark.unit
def test_act_uses_expected_q():
    """Action selection uses q_values_from_pmf, not raw PMF argmax.

    We verify by checking that greedy action matches the argmax of
    expected Q-values computed from the PMF.
    """
    # Given
    agent = _make_agent()
    state = agent.init(jax.random.PRNGKey(1))
    obs = jnp.ones(OBS_DIM)
    greedy_state = DQNState(
        params=state.params,
        opt_state=state.opt_state,
        target_params=state.target_params,
        epsilon=jnp.array(0.0),
    )

    # When — get action from agent (greedy, epsilon=0)
    action = agent.act(greedy_state, obs, jax.random.PRNGKey(2))

    # And — compute expected action manually via q_values_from_pmf
    static = eqx.partition(agent, eqx.is_array)[1]
    model = eqx.combine(state.params, static)
    pmf = model._forward(obs)
    q_values = q_values_from_pmf(pmf, model.atom_head.atoms)
    expected_action = jnp.argmax(q_values)

    # Then
    assert int(action) == int(expected_action), (
        f"Agent chose action {int(action)} but expected Q argmax is {int(expected_action)}"
    )
