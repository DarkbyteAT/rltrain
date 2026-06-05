"""Tests for SquashedNormal distribution — numerical stability and correctness."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from rltrain.distributions import SquashedNormal


@pytest.mark.unit
def test_sample_and_log_prob_consistent():
    """sample_and_log_prob must match sample + log_prob for the same key."""
    # Given
    key = jax.random.PRNGKey(0)
    dist = SquashedNormal(loc=jnp.array([0.5, -0.3]), scale=jnp.array([1.0, 0.5]))

    # When
    action_combined, lp_combined = dist.sample_and_log_prob(key)
    action_separate = dist.sample(key)
    lp_separate = dist.log_prob(action_separate)

    # Then — actions must be identical (same key, same RNG path)
    assert jnp.allclose(action_combined, action_separate)
    # Log probs must be close (combined avoids atanh, separate uses atanh)
    assert jnp.allclose(lp_combined, lp_separate, atol=1e-5)


@pytest.mark.unit
def test_grad_flows_through_sample():
    """jax.grad must flow through sample via the reparameterization trick."""
    # Given
    key = jax.random.PRNGKey(1)

    def loss(loc):
        dist = SquashedNormal(loc=loc, scale=jnp.ones_like(loc))
        action = dist.sample(key)
        return jnp.sum(action)

    # When
    grad = jax.grad(loss)(jnp.array([0.0, 1.0]))

    # Then
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.any(grad != 0.0), "Gradient should flow through sample"


@pytest.mark.unit
def test_log_prob_finite_at_extreme_loc():
    """log_prob must stay finite when loc pushes tanh(u) near +/-1."""
    # Given — loc=5.0 means tanh(u) ~ 1.0, which breaks naive log(1-tanh^2)
    key = jax.random.PRNGKey(2)
    dist = SquashedNormal(loc=jnp.array([5.0, -5.0, 10.0, -10.0]), scale=jnp.ones(4))

    # When
    action = dist.sample(key)
    lp = dist.log_prob(action)

    # Then
    assert jnp.all(jnp.isfinite(lp))
    assert jnp.all(jnp.isfinite(action))


@pytest.mark.unit
def test_log_prob_finite_at_extreme_scale():
    """log_prob must stay finite across very small and very large scale values."""
    # Given
    key = jax.random.PRNGKey(3)

    for scale_val in [0.01, 0.1, 1.0, 10.0]:
        # When
        dist = SquashedNormal(loc=jnp.zeros(2), scale=jnp.full(2, scale_val))
        action = dist.sample(key)
        lp = dist.log_prob(action)

        # Then
        assert jnp.all(jnp.isfinite(lp)), f"NaN at scale={scale_val}"


@pytest.mark.unit
def test_sample_bounded():
    """All samples must lie strictly in (-1, 1) — the tanh range."""
    # Given
    keys = jax.random.split(jax.random.PRNGKey(4), 1000)
    dist = SquashedNormal(loc=jnp.array([3.0, -3.0]), scale=jnp.ones(2))

    # When
    actions = jax.vmap(dist.sample)(keys)

    # Then
    assert jnp.all(actions > -1.0)
    assert jnp.all(actions < 1.0)


@pytest.mark.unit
def test_log_prob_not_unreasonably_large():
    """Log probabilities should not be unreasonably large."""
    # Given
    key = jax.random.PRNGKey(5)
    dist = SquashedNormal(loc=jnp.zeros(3), scale=jnp.ones(3))

    # When
    action = dist.sample(key)
    lp = dist.log_prob(action)

    # Then
    assert jnp.all(lp < 10.0), "log_prob should not be unreasonably large"


@pytest.mark.unit
def test_batch_dimensions():
    """Works with scalar, 1D, and 2D batch shapes."""
    # Given
    key = jax.random.PRNGKey(6)

    # When — scalar
    dist1 = SquashedNormal(loc=jnp.array(0.0), scale=jnp.array(1.0))
    a1 = dist1.sample(key)

    # Then
    assert a1.shape == ()

    # When — 1D
    dist2 = SquashedNormal(loc=jnp.zeros(4), scale=jnp.ones(4))
    a2 = dist2.sample(key)

    # Then
    assert a2.shape == (4,)

    # When — 2D
    dist3 = SquashedNormal(loc=jnp.zeros((3, 2)), scale=jnp.ones((3, 2)))
    a3 = dist3.sample(key)

    # Then
    assert a3.shape == (3, 2)


@pytest.mark.unit
def test_jit_compatible():
    """All methods must work under eqx.filter_jit (Equinox modules need filter_jit, not jax.jit)."""
    import equinox as eqx

    # Given
    key = jax.random.PRNGKey(7)
    dist = SquashedNormal(loc=jnp.zeros(2), scale=jnp.ones(2))

    # When — eqx.filter_jit handles the array/static partition automatically
    a = eqx.filter_jit(lambda d, k: d.sample(k))(dist, key)
    lp = eqx.filter_jit(lambda d, x: d.log_prob(x))(dist, a)
    a2, lp2 = eqx.filter_jit(lambda d, k: d.sample_and_log_prob(k))(dist, key)

    # Then
    assert jnp.all(jnp.isfinite(lp))
    assert jnp.all(jnp.isfinite(lp2))
