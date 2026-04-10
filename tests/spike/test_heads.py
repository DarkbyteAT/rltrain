"""Tests for action head output parameterisations."""

import jax
import jax.numpy as jnp
import pytest

from spike.heads import (
    BetaHead,
    DiscreteHead,
    GammaHead,
    GaussianHead,
    SquashedGaussianHead,
)


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


@pytest.fixture
def features():
    return jnp.ones(64)


def test_discrete_head_categorical(key, features):
    """Given features, DiscreteHead produces a Categorical with correct action dim."""
    head = DiscreteHead(feature_dim=64, action_dim=3, key=key)
    dist = head(features)
    action = dist.sample(key)
    log_prob = dist.log_prob(action)

    assert action.shape == ()
    assert log_prob.shape == ()
    assert 0 <= int(action) < 3


def test_gaussian_head_normal(key, features):
    """Given features, GaussianHead produces a Normal with correct shape."""
    head = GaussianHead(feature_dim=64, action_dim=2, key=key)
    dist = head(features)
    action = dist.sample(key)
    log_prob = dist.log_prob(action)

    assert action.shape == (2,)
    assert log_prob.shape == (2,)


def test_squashed_gaussian_head_bounded(key, features):
    """Given features, SquashedGaussianHead produces actions in (-1, 1)."""
    head = SquashedGaussianHead(feature_dim=64, action_dim=2, key=key)
    dist = head(features)
    action = dist.sample(key)
    log_prob = dist.log_prob(action)

    assert action.shape == (2,)
    assert jnp.all(action > -1.0) and jnp.all(action < 1.0)


def test_gamma_head_positive(key, features):
    """Given features, GammaHead produces non-negative samples."""
    head = GammaHead(feature_dim=64, action_dim=2, key=key)
    dist = head(features)
    action = dist.sample(key)

    assert action.shape == (2,)
    assert jnp.all(action >= 0.0)


def test_beta_head_bounded(key, features):
    """Given features, BetaHead produces samples in (0, 1)."""
    head = BetaHead(feature_dim=64, action_dim=2, key=key)
    dist = head(features)
    action = dist.sample(key)

    assert action.shape == (2,)
    assert jnp.all(action > 0.0) and jnp.all(action < 1.0)


def test_discrete_head_sample_and_log_prob(key, features):
    """Given features, sample_and_log_prob returns consistent action and log_prob."""
    head = DiscreteHead(feature_dim=64, action_dim=5, key=key)
    dist = head(features)
    action, log_prob = dist.sample_and_log_prob(key)

    # Verify consistency
    log_prob_check = dist.log_prob(action)
    assert jnp.allclose(log_prob, log_prob_check)


def test_heads_are_jittable(key, features):
    """All heads work under jax.jit."""
    for HeadClass in [DiscreteHead, GaussianHead, SquashedGaussianHead]:
        head = HeadClass(feature_dim=64, action_dim=2, key=key)

        @jax.jit
        def forward(features, key):
            dist = head(features)
            return dist.sample_and_log_prob(key)

        action, log_prob = forward(features, key)
        assert action.shape is not None
