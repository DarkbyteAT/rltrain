"""Tests for EpochTerminator protocol and KLEarlyStop integration with PPO."""

import jax
import jax.numpy as jnp
import optax
import pytest

from rltrain.agents.ppo import PPO
from rltrain.agents.ppo_terminators import EpochTerminator, KLEarlyStop
from rltrain.heads import DiscreteHead
from rltrain.networks import MLP
from rltrain.transitions import Transition
from tests.agents._helpers import HIDDEN, MINIBATCH, NUM_ACTIONS, OBS_DIM
from tests.agents._helpers import _make_on_policy_transitions as _make_transitions


@pytest.mark.unit
def test_klearlystop_satisfies_epoch_terminator_protocol():
    """KLEarlyStop is structurally a valid EpochTerminator."""
    # Given
    cb = KLEarlyStop(kl_threshold=0.05)

    # Then
    assert isinstance(cb, EpochTerminator)


@pytest.mark.unit
def test_klearlystop_does_not_stop_below_threshold():
    """KLEarlyStop returns False when approx_kl is below the threshold."""
    # Given
    cb = KLEarlyStop(kl_threshold=0.05)
    metrics = {"approx_kl": jnp.array(0.01)}

    # When
    stop = cb.should_stop(metrics)

    # Then
    assert not bool(stop)


@pytest.mark.unit
def test_klearlystop_stops_above_threshold():
    """KLEarlyStop returns True when approx_kl exceeds the threshold."""
    # Given
    cb = KLEarlyStop(kl_threshold=0.05)
    metrics = {"approx_kl": jnp.array(0.5)}

    # When
    stop = cb.should_stop(metrics)

    # Then
    assert bool(stop)


@pytest.mark.unit
def test_ppo_runs_with_terminator_attached():
    """PPO learn step still produces a finite loss when a KLEarlyStop terminator is attached."""
    # Given
    key = jax.random.PRNGKey(0)
    k_actor, k_critic, k_head, k_learn = jax.random.split(key, 4)
    actor = MLP(in_size=OBS_DIM, out_size=HIDDEN, width=HIDDEN, depth=1, key=k_actor)
    critic = MLP(in_size=OBS_DIM, out_size=1, width=HIDDEN, depth=1, key=k_critic)
    head = DiscreteHead(feature_dim=HIDDEN, action_dim=NUM_ACTIONS, key=k_head)
    agent = PPO(
        actor=actor,
        critic=critic,
        action_head=head,
        optimizer=optax.adam(3e-4),
        gamma=0.99,
        tau=0.0,
        beta_critic=0.5,
        lambda_gae=0.95,
        eps_clip=0.2,
        num_epochs=4,
        minibatch_size=MINIBATCH,
        epoch_terminators=(KLEarlyStop(kl_threshold=0.01),),
    )

    state = agent.init(key)
    transitions: Transition = _make_transitions(k_learn)

    # When
    _new_state, metrics = agent.learn(state, transitions, k_learn)

    # Then
    assert jnp.isfinite(metrics["loss"])
    assert "approx_kl" in metrics


def _make_ppo(key, *, num_epochs: int, epoch_terminators=()):
    """Build a small PPO with a tunable epoch count and terminator tuple."""
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
        num_epochs=num_epochs,
        minibatch_size=MINIBATCH,
        epoch_terminators=epoch_terminators,
    )


@pytest.mark.unit
def test_terminator_mask_freezes_remaining_epochs():
    """A ``kl_threshold=-inf`` terminator fires after epoch 1; epochs 2..N must be masked.

    Builds two PPOs over the same data, key, and init weights: one with
    ``num_epochs=1`` and no terminator, one with ``num_epochs=4`` and a
    ``KLEarlyStop(kl_threshold=-inf)``. ``-inf`` makes the comparison
    ``approx_kl > -inf`` always True regardless of sign — Schulman's
    approximate KL estimator (``mean(log_pi_old - log_pi_new)``) can be
    negative on the first epoch when the policy update happens to
    increase log-probs of taken actions. Using ``-inf`` sidesteps that
    quirk so the terminator fires deterministically.

    If the ``stopped``-flag masking in the scan body works, the 4-epoch
    path freezes after epoch 1 — both paths should produce numerically
    identical post-learn params.
    """
    # Given identical PPOs and batches
    init_key = jax.random.PRNGKey(0)
    batch_key = jax.random.PRNGKey(1)
    learn_key = jax.random.PRNGKey(2)

    agent_short = _make_ppo(init_key, num_epochs=1, epoch_terminators=())
    agent_long = _make_ppo(
        init_key,
        num_epochs=4,
        epoch_terminators=(KLEarlyStop(kl_threshold=float("-inf")),),
    )

    state_short = agent_short.init(init_key)
    state_long = agent_long.init(init_key)

    # Sanity: both agents start with identical params
    for s, t in zip(jax.tree.leaves(state_short.params), jax.tree.leaves(state_long.params), strict=True):
        assert jnp.array_equal(s, t), "Same init seed should produce identical initial params"

    batch = _make_transitions(batch_key, n=HIDDEN * 2)

    # When both run one learn step
    new_state_short, _ = agent_short.learn(state_short, batch, learn_key)
    new_state_long, metrics_long = agent_long.learn(state_long, batch, learn_key)

    # Then post-learn params match (epochs 2..4 frozen by the terminator)
    leaves_short = jax.tree.leaves(new_state_short.params)
    leaves_long = jax.tree.leaves(new_state_long.params)
    assert len(leaves_short) == len(leaves_long), "Param pytree structure must match"
    for short, long in zip(leaves_short, leaves_long, strict=True):
        assert jnp.allclose(short, long, atol=1e-6), (
            f"Terminator mask leaked: max diff {float(jnp.max(jnp.abs(short - long)))}"
        )

    # And the long path's metrics are still finite (loss accumulator was masked, not corrupted)
    assert jnp.isfinite(metrics_long["loss"])
    assert jnp.isfinite(metrics_long["approx_kl"])
