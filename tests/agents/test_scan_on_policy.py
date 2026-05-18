"""Tests that every on-policy agent works as a lax.scan carry.

Each agent's ``learn(state, batch, key)`` must compose as the body of
``jax.lax.scan`` where ``state`` is the carry and ``batch`` is the scanned
input.  This is the fundamental requirement for vectorised training loops.
"""

import jax
import jax.numpy as jnp
import optax
import pytest

from rltrain.agents.advantage_ac import AdvantageAC
from rltrain.agents.ppo import PPO
from rltrain.agents.reinforce import REINFORCE
from rltrain.agents.spo import SPO
from rltrain.agents.vanilla_ac import VanillaAC
from rltrain.agents.vanilla_pg import VanillaPG
from rltrain.heads import DiscreteHead
from rltrain.networks import MLP
from rltrain.transitions import Transition
from tests.agents._helpers import HIDDEN, MINIBATCH, NUM_ACTIONS, OBS_DIM


K = 3  # number of scan steps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stacked_batches(key, batch_size):
    """Create K batches stacked along leading axis for scanning."""
    keys = jax.random.split(key, K)

    def make_one(k):
        k1, k2, k3 = jax.random.split(k, 3)
        return Transition(
            obs=jax.random.normal(k1, (batch_size, OBS_DIM)),
            action=jax.random.randint(k2, (batch_size,), 0, NUM_ACTIONS),
            reward=jax.random.normal(k3, (batch_size,)),
            next_obs=jax.random.normal(k1, (batch_size, OBS_DIM)),
            done=jnp.zeros(batch_size, dtype=jnp.bool_),
            log_prob=jnp.zeros(batch_size),
            value=jnp.zeros(batch_size),
        )

    return jax.vmap(make_one)(keys)


def _run_scan(agent, state, batches):
    """Run lax.scan over K learn steps and return (final_state, all_metrics)."""
    keys = jax.random.split(jax.random.PRNGKey(99), K)

    def step(carry, xs):
        st, _carry_key = carry
        batch, step_key = xs
        new_st, metrics = agent.learn(st, batch, step_key)
        return (new_st, _carry_key), metrics

    init_carry = (state, jax.random.PRNGKey(0))
    (final_state, _), all_metrics = jax.lax.scan(step, init_carry, (batches, keys))
    return final_state, all_metrics


def _params_differ(params_a, params_b) -> bool:
    """Check that at least one leaf in the parameter pytree has changed."""
    leaves_a = jax.tree.leaves(params_a)
    leaves_b = jax.tree.leaves(params_b)
    return any(not jnp.array_equal(a, b) for a, b in zip(leaves_a, leaves_b, strict=True))


# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------


def _make_vanilla_pg(key):
    k1, k2 = jax.random.split(key)
    return VanillaPG(
        actor=MLP(OBS_DIM, HIDDEN, width=HIDDEN, depth=1, key=k1),
        action_head=DiscreteHead(HIDDEN, NUM_ACTIONS, key=k2),
        optimizer=optax.adam(1e-3),
        gamma=0.99,
        tau=0.01,
        normalise=True,
    )


def _make_reinforce(key):
    k1, k2, k3 = jax.random.split(key, 3)
    return REINFORCE(
        actor=MLP(OBS_DIM, HIDDEN, width=HIDDEN, depth=1, key=k1),
        action_head=DiscreteHead(HIDDEN, NUM_ACTIONS, key=k2),
        critic=MLP(OBS_DIM, 1, width=HIDDEN, depth=1, key=k3),
        optimizer=optax.adam(1e-3),
        gamma=0.99,
        tau=0.01,
        beta_critic=0.5,
    )


def _make_vanilla_ac(key):
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


def _make_advantage_ac(key):
    k1, k2, k3 = jax.random.split(key, 3)
    return AdvantageAC(
        actor=MLP(OBS_DIM, HIDDEN, width=HIDDEN, depth=1, key=k1),
        action_head=DiscreteHead(HIDDEN, NUM_ACTIONS, key=k2),
        critic=MLP(OBS_DIM, 1, width=HIDDEN, depth=1, key=k3),
        optimizer=optax.adam(1e-3),
        gamma=0.99,
        tau=0.01,
        beta_critic=0.5,
        lambda_gae=0.95,
    )


def _make_ppo(key):
    k1, k2, k3 = jax.random.split(key, 3)
    return PPO(
        actor=MLP(OBS_DIM, HIDDEN, width=HIDDEN, depth=1, key=k1),
        action_head=DiscreteHead(HIDDEN, NUM_ACTIONS, key=k2),
        critic=MLP(OBS_DIM, 1, width=HIDDEN, depth=1, key=k3),
        optimizer=optax.adam(1e-3),
        gamma=0.99,
        tau=0.01,
        beta_critic=0.5,
        lambda_gae=0.95,
        eps_clip=0.2,
        num_epochs=2,
        minibatch_size=MINIBATCH,
    )


def _make_spo(key):
    k1, k2, k3 = jax.random.split(key, 3)
    return SPO(
        actor=MLP(OBS_DIM, HIDDEN, width=HIDDEN, depth=1, key=k1),
        action_head=DiscreteHead(HIDDEN, NUM_ACTIONS, key=k2),
        critic=MLP(OBS_DIM, 1, width=HIDDEN, depth=1, key=k3),
        optimizer=optax.adam(1e-3),
        gamma=0.99,
        tau=0.01,
        beta_critic=0.5,
        lambda_gae=0.95,
        eps_clip=0.2,
        num_epochs=2,
        minibatch_size=MINIBATCH,
    )


# ---------------------------------------------------------------------------
# VanillaPG
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_vanilla_pg_scan_compiles():
    """Given a VanillaPG agent, lax.scan over K=3 learn steps compiles and runs."""
    # Given
    agent = _make_vanilla_pg(jax.random.PRNGKey(0))
    state = agent.init(jax.random.PRNGKey(1))
    batches = _make_stacked_batches(jax.random.PRNGKey(2), batch_size=16)

    # When
    final_state, all_metrics = _run_scan(agent, state, batches)

    # Then
    assert final_state is not None
    assert "loss" in all_metrics


@pytest.mark.unit
def test_vanilla_pg_scan_params_change():
    """Given a VanillaPG agent, scanning K steps produces different params from init."""
    # Given
    agent = _make_vanilla_pg(jax.random.PRNGKey(0))
    state = agent.init(jax.random.PRNGKey(1))
    batches = _make_stacked_batches(jax.random.PRNGKey(2), batch_size=16)

    # When
    final_state, _ = _run_scan(agent, state, batches)

    # Then
    assert _params_differ(state.params, final_state.params)


@pytest.mark.unit
def test_vanilla_pg_scan_metrics_shape():
    """Given a VanillaPG agent, scan metrics have shape (K,) and are all finite."""
    # Given
    agent = _make_vanilla_pg(jax.random.PRNGKey(0))
    state = agent.init(jax.random.PRNGKey(1))
    batches = _make_stacked_batches(jax.random.PRNGKey(2), batch_size=16)

    # When
    _, all_metrics = _run_scan(agent, state, batches)

    # Then
    assert all_metrics["loss"].shape == (K,)
    assert jnp.all(jnp.isfinite(all_metrics["loss"]))


# ---------------------------------------------------------------------------
# REINFORCE
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_reinforce_scan_compiles():
    """Given a REINFORCE agent, lax.scan over K=3 learn steps compiles and runs."""
    # Given
    agent = _make_reinforce(jax.random.PRNGKey(10))
    state = agent.init(jax.random.PRNGKey(11))
    batches = _make_stacked_batches(jax.random.PRNGKey(12), batch_size=16)

    # When
    final_state, all_metrics = _run_scan(agent, state, batches)

    # Then
    assert final_state is not None
    assert "loss" in all_metrics


@pytest.mark.unit
def test_reinforce_scan_params_change():
    """Given a REINFORCE agent, scanning K steps produces different params from init."""
    # Given
    agent = _make_reinforce(jax.random.PRNGKey(10))
    state = agent.init(jax.random.PRNGKey(11))
    batches = _make_stacked_batches(jax.random.PRNGKey(12), batch_size=16)

    # When
    final_state, _ = _run_scan(agent, state, batches)

    # Then
    assert _params_differ(state.params, final_state.params)


@pytest.mark.unit
def test_reinforce_scan_metrics_shape():
    """Given a REINFORCE agent, scan metrics have shape (K,) and are all finite."""
    # Given
    agent = _make_reinforce(jax.random.PRNGKey(10))
    state = agent.init(jax.random.PRNGKey(11))
    batches = _make_stacked_batches(jax.random.PRNGKey(12), batch_size=16)

    # When
    _, all_metrics = _run_scan(agent, state, batches)

    # Then
    assert all_metrics["loss"].shape == (K,)
    assert jnp.all(jnp.isfinite(all_metrics["loss"]))


# ---------------------------------------------------------------------------
# VanillaAC
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_vanilla_ac_scan_compiles():
    """Given a VanillaAC agent, lax.scan over K=3 learn steps compiles and runs."""
    # Given
    agent = _make_vanilla_ac(jax.random.PRNGKey(20))
    state = agent.init(jax.random.PRNGKey(21))
    batches = _make_stacked_batches(jax.random.PRNGKey(22), batch_size=16)

    # When
    final_state, all_metrics = _run_scan(agent, state, batches)

    # Then
    assert final_state is not None
    assert "loss" in all_metrics


@pytest.mark.unit
def test_vanilla_ac_scan_params_change():
    """Given a VanillaAC agent, scanning K steps produces different params from init."""
    # Given
    agent = _make_vanilla_ac(jax.random.PRNGKey(20))
    state = agent.init(jax.random.PRNGKey(21))
    batches = _make_stacked_batches(jax.random.PRNGKey(22), batch_size=16)

    # When
    final_state, _ = _run_scan(agent, state, batches)

    # Then
    assert _params_differ(state.params, final_state.params)


@pytest.mark.unit
def test_vanilla_ac_scan_metrics_shape():
    """Given a VanillaAC agent, scan metrics have shape (K,) and are all finite."""
    # Given
    agent = _make_vanilla_ac(jax.random.PRNGKey(20))
    state = agent.init(jax.random.PRNGKey(21))
    batches = _make_stacked_batches(jax.random.PRNGKey(22), batch_size=16)

    # When
    _, all_metrics = _run_scan(agent, state, batches)

    # Then
    assert all_metrics["loss"].shape == (K,)
    assert jnp.all(jnp.isfinite(all_metrics["loss"]))


# ---------------------------------------------------------------------------
# AdvantageAC
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_advantage_ac_scan_compiles():
    """Given an AdvantageAC agent, lax.scan over K=3 learn steps compiles and runs."""
    # Given
    agent = _make_advantage_ac(jax.random.PRNGKey(30))
    state = agent.init(jax.random.PRNGKey(31))
    batches = _make_stacked_batches(jax.random.PRNGKey(32), batch_size=16)

    # When
    final_state, all_metrics = _run_scan(agent, state, batches)

    # Then
    assert final_state is not None
    assert "loss" in all_metrics


@pytest.mark.unit
def test_advantage_ac_scan_params_change():
    """Given an AdvantageAC agent, scanning K steps produces different params from init."""
    # Given
    agent = _make_advantage_ac(jax.random.PRNGKey(30))
    state = agent.init(jax.random.PRNGKey(31))
    batches = _make_stacked_batches(jax.random.PRNGKey(32), batch_size=16)

    # When
    final_state, _ = _run_scan(agent, state, batches)

    # Then
    assert _params_differ(state.params, final_state.params)


@pytest.mark.unit
def test_advantage_ac_scan_metrics_shape():
    """Given an AdvantageAC agent, scan metrics have shape (K,) and are all finite."""
    # Given
    agent = _make_advantage_ac(jax.random.PRNGKey(30))
    state = agent.init(jax.random.PRNGKey(31))
    batches = _make_stacked_batches(jax.random.PRNGKey(32), batch_size=16)

    # When
    _, all_metrics = _run_scan(agent, state, batches)

    # Then
    assert all_metrics["loss"].shape == (K,)
    assert jnp.all(jnp.isfinite(all_metrics["loss"]))


# ---------------------------------------------------------------------------
# PPO (horizon-sized batches, nested epoch loop inside learn)
# ---------------------------------------------------------------------------

PPO_BATCH_SIZE = 32  # must be divisible by MINIBATCH (16)


@pytest.mark.unit
def test_ppo_scan_compiles():
    """Given a PPO agent, lax.scan over K=3 learn steps compiles and runs."""
    # Given
    agent = _make_ppo(jax.random.PRNGKey(40))
    state = agent.init(jax.random.PRNGKey(41))
    batches = _make_stacked_batches(jax.random.PRNGKey(42), batch_size=PPO_BATCH_SIZE)

    # When
    final_state, all_metrics = _run_scan(agent, state, batches)

    # Then
    assert final_state is not None
    assert "loss" in all_metrics


@pytest.mark.unit
def test_ppo_scan_params_change():
    """Given a PPO agent, scanning K steps produces different params from init."""
    # Given
    agent = _make_ppo(jax.random.PRNGKey(40))
    state = agent.init(jax.random.PRNGKey(41))
    batches = _make_stacked_batches(jax.random.PRNGKey(42), batch_size=PPO_BATCH_SIZE)

    # When
    final_state, _ = _run_scan(agent, state, batches)

    # Then
    assert _params_differ(state.params, final_state.params)


@pytest.mark.unit
def test_ppo_scan_metrics_shape():
    """Given a PPO agent, scan metrics have shape (K,) and are all finite."""
    # Given
    agent = _make_ppo(jax.random.PRNGKey(40))
    state = agent.init(jax.random.PRNGKey(41))
    batches = _make_stacked_batches(jax.random.PRNGKey(42), batch_size=PPO_BATCH_SIZE)

    # When
    _, all_metrics = _run_scan(agent, state, batches)

    # Then
    assert all_metrics["loss"].shape == (K,)
    assert jnp.all(jnp.isfinite(all_metrics["loss"]))


# ---------------------------------------------------------------------------
# SPO (same structure as PPO — quadratic penalty surrogate)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_spo_scan_compiles():
    """Given an SPO agent, lax.scan over K=3 learn steps compiles and runs."""
    # Given
    agent = _make_spo(jax.random.PRNGKey(50))
    state = agent.init(jax.random.PRNGKey(51))
    batches = _make_stacked_batches(jax.random.PRNGKey(52), batch_size=PPO_BATCH_SIZE)

    # When
    final_state, all_metrics = _run_scan(agent, state, batches)

    # Then
    assert final_state is not None
    assert "loss" in all_metrics


@pytest.mark.unit
def test_spo_scan_params_change():
    """Given an SPO agent, scanning K steps produces different params from init."""
    # Given
    agent = _make_spo(jax.random.PRNGKey(50))
    state = agent.init(jax.random.PRNGKey(51))
    batches = _make_stacked_batches(jax.random.PRNGKey(52), batch_size=PPO_BATCH_SIZE)

    # When
    final_state, _ = _run_scan(agent, state, batches)

    # Then
    assert _params_differ(state.params, final_state.params)


@pytest.mark.unit
def test_spo_scan_metrics_shape():
    """Given an SPO agent, scan metrics have shape (K,) and are all finite."""
    # Given
    agent = _make_spo(jax.random.PRNGKey(50))
    state = agent.init(jax.random.PRNGKey(51))
    batches = _make_stacked_batches(jax.random.PRNGKey(52), batch_size=PPO_BATCH_SIZE)

    # When
    _, all_metrics = _run_scan(agent, state, batches)

    # Then
    assert all_metrics["loss"].shape == (K,)
    assert jnp.all(jnp.isfinite(all_metrics["loss"]))
