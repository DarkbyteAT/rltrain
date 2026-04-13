"""Scannability tests for off-policy agents — every agent.learn must work as a lax.scan body."""

import jax
import jax.numpy as jnp
import optax
import pytest

from spike.agents.distributional_dqn import DistributionalDQN
from spike.agents.double_dqn import DoubleDQN
from spike.agents.sac import SAC, SACState
from spike.agents.vanilla_dqn import DQNState, VanillaDQN
from spike.heads import CategoricalAtomHead, DiscreteHead, SquashedGaussianHead
from spike.networks import MLP
from spike.transitions import Transition


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OBS_DIM = 4
NUM_ACTIONS = 2
ACTION_DIM = 2
BATCH_SIZE = 16
K = 3


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------


def _make_dqn_stacked_batches(key, K, batch_size, obs_dim, num_actions):
    """Create K batches of discrete-action transitions stacked along axis 0."""
    keys = jax.random.split(key, K)

    def make_one(k):
        k1, k2, k3, k4 = jax.random.split(k, 4)
        return Transition(
            obs=jax.random.normal(k1, (batch_size, obs_dim)),
            action=jax.random.randint(k2, (batch_size,), 0, num_actions),
            reward=jax.random.normal(k3, (batch_size,)),
            next_obs=jax.random.normal(k4, (batch_size, obs_dim)),
            done=jnp.zeros(batch_size, dtype=jnp.bool_),
            log_prob=jnp.zeros(batch_size),
            value=jnp.zeros(batch_size),
        )

    return jax.vmap(make_one)(keys)


def _make_sac_continuous_stacked_batches(key, K, batch_size, obs_dim, action_dim):
    """Create K batches of continuous-action transitions stacked along axis 0."""
    keys = jax.random.split(key, K)

    def make_one(k):
        k1, k2, k3, k4 = jax.random.split(k, 4)
        return Transition(
            obs=jax.random.normal(k1, (batch_size, obs_dim)),
            action=jax.random.normal(k2, (batch_size, action_dim)),
            reward=jax.random.normal(k3, (batch_size,)),
            next_obs=jax.random.normal(k4, (batch_size, obs_dim)),
            done=jnp.zeros(batch_size, dtype=jnp.bool_),
            log_prob=jnp.zeros(batch_size),
            value=jnp.zeros(batch_size),
        )

    return jax.vmap(make_one)(keys)


# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------


def _make_vanilla_dqn(key):
    return VanillaDQN(
        q_net=MLP(OBS_DIM, NUM_ACTIONS, width=64, depth=1, key=key),
        optimizer=optax.adam(1e-3),
        gamma=0.99,
        target_rate=0.01,
        num_actions=NUM_ACTIONS,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.01,
    )


def _make_double_dqn(key):
    return DoubleDQN(
        q_net=MLP(OBS_DIM, NUM_ACTIONS, width=64, depth=1, key=key),
        optimizer=optax.adam(1e-3),
        gamma=0.99,
        target_rate=0.01,
        num_actions=NUM_ACTIONS,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.01,
    )


def _make_c51(key):
    k1, k2 = jax.random.split(key)
    return DistributionalDQN(
        feature_net=MLP(OBS_DIM, 64, width=64, depth=1, key=k1),
        atom_head=CategoricalAtomHead(64, NUM_ACTIONS, num_atoms=11, v_min=-5, v_max=5, key=k2),
        optimizer=optax.adam(1e-3),
        gamma=0.99,
        target_rate=0.01,
        num_actions=NUM_ACTIONS,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.01,
    )


def _make_continuous_sac(key):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return SAC(
        actor=MLP(OBS_DIM, 64, width=64, depth=1, key=k1),
        action_head=SquashedGaussianHead(64, ACTION_DIM, key=k2),
        critic_1=MLP(OBS_DIM + ACTION_DIM, 1, width=64, depth=1, key=k3),
        critic_2=MLP(OBS_DIM + ACTION_DIM, 1, width=64, depth=1, key=k4),
        actor_optimizer=optax.adam(3e-4),
        critic_optimizer=optax.adam(3e-4),
        alpha_optimizer=optax.adam(3e-4),
        gamma=0.99,
        tau=0.005,
    )


def _make_discrete_sac(key):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return SAC(
        actor=MLP(OBS_DIM, 64, width=64, depth=1, key=k1),
        action_head=DiscreteHead(64, NUM_ACTIONS, key=k2),
        critic_1=MLP(OBS_DIM, NUM_ACTIONS, width=64, depth=1, key=k3),
        critic_2=MLP(OBS_DIM, NUM_ACTIONS, width=64, depth=1, key=k4),
        actor_optimizer=optax.adam(3e-4),
        critic_optimizer=optax.adam(3e-4),
        alpha_optimizer=optax.adam(3e-4),
        gamma=0.99,
        tau=0.005,
    )


# ---------------------------------------------------------------------------
# Scan runner helpers
# ---------------------------------------------------------------------------


def _run_dqn_scan(agent):
    """Initialise a DQN-family agent and run K learn steps via lax.scan."""
    key = jax.random.PRNGKey(42)
    k_init, k_batch, k_steps = jax.random.split(key, 3)

    state = agent.init(k_init)
    batches = _make_dqn_stacked_batches(k_batch, K, BATCH_SIZE, OBS_DIM, NUM_ACTIONS)
    step_keys = jax.random.split(k_steps, K)

    def step(carry, xs):
        st, _ = carry
        batch, step_key = xs
        new_st, metrics = agent.learn(st, batch, step_key)
        return (new_st, step_key), metrics

    (final_state, _), all_metrics = jax.lax.scan(step, (state, k_init), (batches, step_keys))
    return state, final_state, all_metrics


def _run_sac_scan(agent, continuous):
    """Initialise a SAC agent and run K learn steps via lax.scan."""
    key = jax.random.PRNGKey(42)
    k_init, k_batch, k_steps = jax.random.split(key, 3)

    state = agent.init(k_init)
    if continuous:
        batches = _make_sac_continuous_stacked_batches(k_batch, K, BATCH_SIZE, OBS_DIM, ACTION_DIM)
    else:
        batches = _make_dqn_stacked_batches(k_batch, K, BATCH_SIZE, OBS_DIM, NUM_ACTIONS)
    step_keys = jax.random.split(k_steps, K)

    def step(carry, xs):
        st, _ = carry
        batch, step_key = xs
        new_st, metrics = agent.learn(st, batch, step_key)
        return (new_st, step_key), metrics

    (final_state, _), all_metrics = jax.lax.scan(step, (state, k_init), (batches, step_keys))
    return state, final_state, all_metrics


# ===========================================================================
# VanillaDQN
# ===========================================================================


@pytest.mark.unit
def test_vanilla_dqn_scan_compiles():
    """lax.scan over VanillaDQN.learn compiles and runs without error."""
    # Given
    agent = _make_vanilla_dqn(jax.random.PRNGKey(0))

    # When
    _state, final_state, all_metrics = _run_dqn_scan(agent)

    # Then
    assert isinstance(final_state, DQNState)
    assert "loss" in all_metrics


@pytest.mark.unit
def test_vanilla_dqn_scan_params_change():
    """VanillaDQN params differ from initial state after K scan steps."""
    # Given
    agent = _make_vanilla_dqn(jax.random.PRNGKey(0))

    # When
    init_state, final_state, _ = _run_dqn_scan(agent)

    # Then
    init_flat = jax.tree.leaves(init_state.params)
    final_flat = jax.tree.leaves(final_state.params)
    any_changed = any(not jnp.allclose(a, b) for a, b in zip(init_flat, final_flat, strict=True))
    assert any_changed, "params should change after K learn steps"


@pytest.mark.unit
def test_vanilla_dqn_scan_metrics_shape():
    """VanillaDQN metrics from scan have shape (K,) and are finite."""
    # Given
    agent = _make_vanilla_dqn(jax.random.PRNGKey(0))

    # When
    _, _, all_metrics = _run_dqn_scan(agent)

    # Then
    assert all_metrics["loss"].shape == (K,)
    assert jnp.all(jnp.isfinite(all_metrics["loss"]))


# ===========================================================================
# DoubleDQN
# ===========================================================================


@pytest.mark.unit
def test_double_dqn_scan_compiles():
    """lax.scan over DoubleDQN.learn compiles and runs without error."""
    # Given
    agent = _make_double_dqn(jax.random.PRNGKey(1))

    # When
    _state, final_state, all_metrics = _run_dqn_scan(agent)

    # Then
    assert isinstance(final_state, DQNState)
    assert "loss" in all_metrics


@pytest.mark.unit
def test_double_dqn_scan_params_change():
    """DoubleDQN params differ from initial state after K scan steps."""
    # Given
    agent = _make_double_dqn(jax.random.PRNGKey(1))

    # When
    init_state, final_state, _ = _run_dqn_scan(agent)

    # Then
    init_flat = jax.tree.leaves(init_state.params)
    final_flat = jax.tree.leaves(final_state.params)
    any_changed = any(not jnp.allclose(a, b) for a, b in zip(init_flat, final_flat, strict=True))
    assert any_changed, "params should change after K learn steps"


@pytest.mark.unit
def test_double_dqn_scan_metrics_shape():
    """DoubleDQN metrics from scan have shape (K,) and are finite."""
    # Given
    agent = _make_double_dqn(jax.random.PRNGKey(1))

    # When
    _, _, all_metrics = _run_dqn_scan(agent)

    # Then
    assert all_metrics["loss"].shape == (K,)
    assert jnp.all(jnp.isfinite(all_metrics["loss"]))


# ===========================================================================
# DistributionalDQN (C51)
# ===========================================================================


@pytest.mark.unit
def test_distributional_dqn_scan_compiles():
    """lax.scan over DistributionalDQN.learn compiles and runs without error."""
    # Given
    agent = _make_c51(jax.random.PRNGKey(2))

    # When
    _state, final_state, all_metrics = _run_dqn_scan(agent)

    # Then
    assert isinstance(final_state, DQNState)
    assert "loss" in all_metrics


@pytest.mark.unit
def test_distributional_dqn_scan_params_change():
    """DistributionalDQN params differ from initial state after K scan steps."""
    # Given
    agent = _make_c51(jax.random.PRNGKey(2))

    # When
    init_state, final_state, _ = _run_dqn_scan(agent)

    # Then
    init_flat = jax.tree.leaves(init_state.params)
    final_flat = jax.tree.leaves(final_state.params)
    any_changed = any(not jnp.allclose(a, b) for a, b in zip(init_flat, final_flat, strict=True))
    assert any_changed, "params should change after K learn steps"


@pytest.mark.unit
def test_distributional_dqn_scan_metrics_shape():
    """DistributionalDQN metrics from scan have shape (K,) and are finite."""
    # Given
    agent = _make_c51(jax.random.PRNGKey(2))

    # When
    _, _, all_metrics = _run_dqn_scan(agent)

    # Then
    assert all_metrics["loss"].shape == (K,)
    assert jnp.all(jnp.isfinite(all_metrics["loss"]))


# ===========================================================================
# SAC continuous
# ===========================================================================


@pytest.mark.unit
def test_sac_continuous_scan_compiles():
    """lax.scan over continuous SAC.learn compiles and runs without error."""
    # Given
    agent = _make_continuous_sac(jax.random.PRNGKey(3))

    # When
    _state, final_state, all_metrics = _run_sac_scan(agent, continuous=True)

    # Then
    assert isinstance(final_state, SACState)
    assert "critic_loss" in all_metrics
    assert "actor_loss" in all_metrics
    assert "alpha_loss" in all_metrics


@pytest.mark.unit
def test_sac_continuous_scan_params_change():
    """Continuous SAC actor and critic params differ after K scan steps."""
    # Given
    agent = _make_continuous_sac(jax.random.PRNGKey(3))

    # When
    init_state, final_state, _ = _run_sac_scan(agent, continuous=True)

    # Then — check both actor and critic params moved
    actor_init = jax.tree.leaves(init_state.actor_params)
    actor_final = jax.tree.leaves(final_state.actor_params)
    actor_changed = any(not jnp.allclose(a, b) for a, b in zip(actor_init, actor_final, strict=True))

    critic_init = jax.tree.leaves(init_state.critic_params)
    critic_final = jax.tree.leaves(final_state.critic_params)
    critic_changed = any(not jnp.allclose(a, b) for a, b in zip(critic_init, critic_final, strict=True))

    assert actor_changed, "actor params should change after K learn steps"
    assert critic_changed, "critic params should change after K learn steps"


@pytest.mark.unit
def test_sac_continuous_scan_metrics_shape():
    """Continuous SAC metrics from scan have shape (K,) and are finite."""
    # Given
    agent = _make_continuous_sac(jax.random.PRNGKey(3))

    # When
    _, _, all_metrics = _run_sac_scan(agent, continuous=True)

    # Then
    for name in ("critic_loss", "actor_loss", "alpha_loss"):
        assert all_metrics[name].shape == (K,), f"{name} should have shape ({K},)"
        assert jnp.all(jnp.isfinite(all_metrics[name])), f"{name} should be finite"


# ===========================================================================
# SAC discrete
# ===========================================================================


@pytest.mark.unit
def test_sac_discrete_scan_compiles():
    """lax.scan over discrete SAC.learn compiles and runs without error."""
    # Given
    agent = _make_discrete_sac(jax.random.PRNGKey(4))

    # When
    _state, final_state, all_metrics = _run_sac_scan(agent, continuous=False)

    # Then
    assert isinstance(final_state, SACState)
    assert "critic_loss" in all_metrics
    assert "actor_loss" in all_metrics
    assert "alpha_loss" in all_metrics


@pytest.mark.unit
def test_sac_discrete_scan_params_change():
    """Discrete SAC actor and critic params differ after K scan steps."""
    # Given
    agent = _make_discrete_sac(jax.random.PRNGKey(4))

    # When
    init_state, final_state, _ = _run_sac_scan(agent, continuous=False)

    # Then
    actor_init = jax.tree.leaves(init_state.actor_params)
    actor_final = jax.tree.leaves(final_state.actor_params)
    actor_changed = any(not jnp.allclose(a, b) for a, b in zip(actor_init, actor_final, strict=True))

    critic_init = jax.tree.leaves(init_state.critic_params)
    critic_final = jax.tree.leaves(final_state.critic_params)
    critic_changed = any(not jnp.allclose(a, b) for a, b in zip(critic_init, critic_final, strict=True))

    assert actor_changed, "actor params should change after K learn steps"
    assert critic_changed, "critic params should change after K learn steps"


@pytest.mark.unit
def test_sac_discrete_scan_metrics_shape():
    """Discrete SAC metrics from scan have shape (K,) and are finite."""
    # Given
    agent = _make_discrete_sac(jax.random.PRNGKey(4))

    # When
    _, _, all_metrics = _run_sac_scan(agent, continuous=False)

    # Then
    for name in ("critic_loss", "actor_loss", "alpha_loss"):
        assert all_metrics[name].shape == (K,), f"{name} should have shape ({K},)"
        assert jnp.all(jnp.isfinite(all_metrics[name])), f"{name} should be finite"
