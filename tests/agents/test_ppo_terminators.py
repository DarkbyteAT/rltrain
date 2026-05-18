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
