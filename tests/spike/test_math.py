"""Tests for RL math primitives — discount, GAE, and distributional RL helpers."""

import jax
import jax.numpy as jnp
import pytest

from spike.math import discount, gae, project_distribution, q_values_from_pmf


# ===========================================================================
# discount
# ===========================================================================


@pytest.mark.unit
def test_discount_basic():
    """Constant rewards with no dones produce the geometric series."""
    # Given
    rewards = jnp.ones(5)
    dones = jnp.zeros(5)

    # When
    returns = discount(rewards, dones, gamma=0.5)

    # Then — G_0 = 1 + 0.5 + 0.25 + 0.125 + 0.0625 = 1.9375
    assert jnp.allclose(returns[0], 1.9375, atol=1e-5)
    assert jnp.allclose(returns[-1], 1.0, atol=1e-5)


@pytest.mark.unit
def test_discount_resets_at_done():
    """Done flag mid-sequence resets the accumulator."""
    # Given
    rewards = jnp.array([1.0, 1.0, 1.0, 1.0])
    dones = jnp.array([0.0, 1.0, 0.0, 0.0])

    # When
    returns = discount(rewards, dones, gamma=0.99)

    # Then — t=1 is a boundary, so G_0 sees only r_0 + γ·r_1·0 = 1+0.99
    assert jnp.allclose(returns[0], 1.99, atol=1e-5)
    assert jnp.allclose(returns[1], 1.0, atol=1e-5)


# ===========================================================================
# GAE
# ===========================================================================


@pytest.mark.unit
def test_gae_returns_correct_shapes():
    """GAE returns (advantages, returns) both of length T when given T+1 values."""
    # Given
    T = 10
    values = jnp.ones(T + 1)
    rewards = jnp.ones(T)
    dones = jnp.zeros(T)

    # When
    advantages, returns = gae(values, rewards, dones, gamma=0.99, lambda_gae=0.95)

    # Then
    assert advantages.shape == (T,)
    assert returns.shape == (T,)


@pytest.mark.unit
def test_gae_with_lambda_one_matches_discount():
    """When lambda=1, GAE advantages equal discounted returns minus values."""
    # Given — random rewards, no dones, V=0 everywhere (so advantages = returns)
    T = 20
    key = jax.random.PRNGKey(42)
    rewards = jax.random.normal(key, (T,))
    dones = jnp.zeros(T)
    values = jnp.zeros(T + 1)  # V(s)=0 → advantages = returns
    gamma = 0.99

    # When
    advantages, returns = gae(values, rewards, dones, gamma, lambda_gae=1.0)
    discounted = discount(rewards, dones, gamma)

    # Then — with V=0 and λ=1, GAE advantages == discount returns
    assert jnp.allclose(advantages, discounted, atol=1e-5)
    assert jnp.allclose(returns, discounted, atol=1e-5)


@pytest.mark.unit
def test_gae_returns_are_advantages_plus_values():
    """Returns = advantages + V(s_t) by definition."""
    # Given
    T = 10
    key = jax.random.PRNGKey(7)
    k1, k2 = jax.random.split(key)
    values = jax.random.normal(k1, (T + 1,))
    rewards = jax.random.normal(k2, (T,))
    dones = jnp.zeros(T)

    # When
    advantages, returns = gae(values, rewards, dones, gamma=0.99, lambda_gae=0.95)

    # Then
    assert jnp.allclose(returns, advantages + values[:-1], atol=1e-6)


@pytest.mark.unit
def test_gae_resets_at_done():
    """Done flags reset the GAE accumulator."""
    # Given — done at t=2, so advantage at t=0 should not see rewards past t=2
    values = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])  # T+1 = 5
    rewards = jnp.array([1.0, 1.0, 1.0, 1.0])  # T = 4
    dones = jnp.array([0.0, 0.0, 1.0, 0.0])  # done at t=2

    # When
    advantages, _ = gae(values, rewards, dones, gamma=0.99, lambda_gae=0.95)

    # Then — advantage at t=3 only sees r_3 (isolated after done at t=2)
    delta_3 = rewards[3] + 0.99 * values[4] - values[3]
    assert jnp.allclose(advantages[3], delta_3, atol=1e-5)


# ===========================================================================
# Distributional RL (C51)
# ===========================================================================


@pytest.mark.unit
def test_q_values_from_pmf_is_expectation():
    """q_values_from_pmf computes the dot product of PMF and atoms."""
    # Given — deterministic distribution: all mass on atom 5.0
    atoms = jnp.array([0.0, 5.0, 10.0])
    pmf = jnp.array([[0.0, 1.0, 0.0], [0.5, 0.0, 0.5]])  # (2 actions, 3 atoms)

    # When
    q = q_values_from_pmf(pmf, atoms)

    # Then
    assert jnp.allclose(q[0], 5.0)
    assert jnp.allclose(q[1], 5.0)  # 0.5*0 + 0.5*10 = 5


@pytest.mark.unit
def test_project_distribution_preserves_mass():
    """Projected PMF sums to 1 along the atom axis."""
    # Given
    num_atoms = 51
    atoms = jnp.linspace(-10.0, 10.0, num_atoms)
    key = jax.random.PRNGKey(0)
    target_pmf = jax.nn.softmax(jax.random.normal(key, (8, num_atoms)), axis=-1)
    rewards = jnp.ones(8)
    dones = jnp.zeros(8)

    # When
    projected = project_distribution(target_pmf, rewards, dones, gamma=0.99, atoms=atoms)

    # Then — each row sums to 1
    row_sums = jnp.sum(projected, axis=-1)
    assert jnp.allclose(row_sums, 1.0, atol=1e-5)


@pytest.mark.unit
def test_project_distribution_done_ignores_discount():
    """When done=1, projected atoms are just the reward (no discount)."""
    # Given — single transition, done, reward=3.0
    atoms = jnp.linspace(-10.0, 10.0, 51)
    target_pmf = jnp.ones((1, 51)) / 51  # uniform
    rewards = jnp.array([3.0])
    dones = jnp.array([1.0])

    # When
    projected = project_distribution(target_pmf, rewards, dones, gamma=0.99, atoms=atoms)

    # Then — all mass should be concentrated at the atom nearest to r=3.0
    q = q_values_from_pmf(projected, atoms)
    assert jnp.allclose(q[0], 3.0, atol=0.5)  # expectation ≈ reward
