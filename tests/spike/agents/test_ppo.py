"""Tests for the PPO agent with clipped surrogate objective."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from spike.agents.agent import Agent
from spike.agents.ppo import PPO
from spike.heads import DiscreteHead
from spike.networks import MLP
from tests.spike.agents._helpers import HIDDEN, MINIBATCH, NUM_ACTIONS, OBS_DIM
from tests.spike.agents._helpers import _make_on_policy_transitions as _make_transitions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(key: jax.Array) -> PPO:
    """Build a small PPO agent for CartPole-sized problems."""
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


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_loss_is_scalar():
    """Given a mini-batch, the PPO loss returns a finite scalar."""
    # Given
    agent = _make_agent(jax.random.PRNGKey(42))
    transitions = _make_transitions(jax.random.PRNGKey(1), n=MINIBATCH)

    # Compute old log-probs
    features = jax.vmap(agent.actor)(transitions.obs)
    dists = jax.vmap(agent.action_head)(features)
    old_log_probs = dists.log_prob(transitions.action)
    advantages = jnp.ones(MINIBATCH)
    returns = jnp.ones(MINIBATCH)

    # When
    loss_val = agent._ppo_loss(transitions, old_log_probs, advantages, returns)

    # Then
    assert loss_val.shape == ()
    assert jnp.isfinite(loss_val)


@pytest.mark.unit
def test_gradients_flow():
    """Gradients through the PPO loss are non-zero."""
    # Given
    agent = _make_agent(jax.random.PRNGKey(7))
    transitions = _make_transitions(jax.random.PRNGKey(2), n=MINIBATCH)

    features = jax.vmap(agent.actor)(transitions.obs)
    dists = jax.vmap(agent.action_head)(features)
    old_log_probs = jax.lax.stop_gradient(dists.log_prob(transitions.action))
    advantages = jnp.ones(MINIBATCH)
    returns = jnp.ones(MINIBATCH)

    # When
    _loss, grads = eqx.filter_value_and_grad(lambda m: m._ppo_loss(transitions, old_log_probs, advantages, returns))(
        agent
    )

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
    new_state, metrics = agent.learn(state, transitions, jax.random.PRNGKey(0))

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
    """PPO satisfies the Agent protocol via structural subtyping."""
    # Given
    agent = _make_agent(jax.random.PRNGKey(0))

    # Then
    assert isinstance(agent, Agent)


@pytest.mark.unit
def test_advantages_are_stop_gradiented():
    """With beta_critic=0, critic should receive zero gradients because
    advantages are stop-gradiented in the actor loss."""
    # Given
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    agent = PPO(
        actor=MLP(OBS_DIM, HIDDEN, width=HIDDEN, depth=1, key=k1),
        action_head=DiscreteHead(HIDDEN, NUM_ACTIONS, key=k2),
        critic=MLP(OBS_DIM, 1, width=HIDDEN, depth=1, key=k3),
        optimizer=optax.adam(1e-3),
        gamma=0.99,
        tau=0.01,
        beta_critic=0.0,
        lambda_gae=0.95,
        eps_clip=0.2,
        num_epochs=1,
        minibatch_size=MINIBATCH,
    )
    transitions = _make_transitions(jax.random.PRNGKey(1), n=MINIBATCH)

    features = jax.vmap(agent.actor)(transitions.obs)
    dists = jax.vmap(agent.action_head)(features)
    old_log_probs = jax.lax.stop_gradient(dists.log_prob(transitions.action))
    advantages = jax.lax.stop_gradient(jnp.ones(MINIBATCH))
    returns = jax.lax.stop_gradient(jnp.ones(MINIBATCH))

    # When
    _loss, grads = eqx.filter_value_and_grad(lambda m: m._ppo_loss(transitions, old_log_probs, advantages, returns))(
        agent
    )

    # Then
    critic_grad_leaves = jax.tree.leaves(eqx.filter(grads.critic, eqx.is_array))
    all_zero = all(jnp.allclose(g, 0.0) for g in critic_grad_leaves)
    assert all_zero, "Critic has non-zero gradients with beta_critic=0"


@pytest.mark.unit
def test_clipped_ratio_bounds():
    """The clipped ratio is bounded by [1-eps, 1+eps].

    We verify indirectly: when old_log_probs diverge significantly from
    current log_probs, the clipped loss should differ from the unclipped loss.
    """
    # Given
    agent = _make_agent(jax.random.PRNGKey(42))
    transitions = _make_transitions(jax.random.PRNGKey(1), n=MINIBATCH)

    # Create deliberately stale old_log_probs (shifted by a large amount)
    features = jax.vmap(agent.actor)(transitions.obs)
    dists = jax.vmap(agent.action_head)(features)
    current_log_probs = dists.log_prob(transitions.action)
    # Shift old_log_probs so ratio = exp(current - old) is far from 1
    old_log_probs = current_log_probs - 2.0
    advantages = jnp.ones(MINIBATCH)
    returns = jnp.ones(MINIBATCH)

    # When — compute PPO loss (clipped) and an unclipped version
    ppo_loss = agent._ppo_loss(transitions, old_log_probs, advantages, returns)

    # Unclipped: just ratio * advantages
    ratio = jnp.exp(current_log_probs - old_log_probs)
    unclipped_actor_loss = -jnp.mean(ratio * advantages)

    # Then — PPO's clipping should make the loss different from raw ratio * A
    # (because ratio = exp(2) ~ 7.4 is well outside [0.8, 1.2])
    assert not jnp.allclose(ppo_loss, unclipped_actor_loss, atol=1e-3), (
        "PPO loss equals unclipped loss despite extreme ratio — clipping may not be active"
    )
