"""Tests for the SPO agent with quadratic penalty surrogate."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from spike.agents.agent import Agent
from spike.agents.ppo import PPO
from spike.agents.spo import SPO
from spike.heads import DiscreteHead
from spike.networks import MLP
from spike.transitions import Transition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OBS_DIM = 4
NUM_ACTIONS = 2
HIDDEN = 32
HORIZON = 32
MINIBATCH = 16


def _make_agent(key: jax.Array) -> SPO:
    """Build a small SPO agent for CartPole-sized problems."""
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


def _make_transitions(key: jax.Array, n: int = HORIZON) -> Transition:
    """Fabricate a horizon batch of random transitions."""
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
    """Given a mini-batch, the SPO loss returns a finite scalar."""
    # Given
    agent = _make_agent(jax.random.PRNGKey(42))
    transitions = _make_transitions(jax.random.PRNGKey(1), n=MINIBATCH)

    features = jax.vmap(agent.actor)(transitions.obs)
    dists = jax.vmap(agent.action_head)(features)
    old_log_probs = dists.log_prob(transitions.action.squeeze(-1))
    advantages = jnp.ones(MINIBATCH)
    returns = jnp.ones(MINIBATCH)

    # When
    loss_val = agent._spo_loss(transitions, old_log_probs, advantages, returns)

    # Then
    assert loss_val.shape == ()
    assert jnp.isfinite(loss_val)


@pytest.mark.unit
def test_gradients_flow():
    """Gradients through the SPO loss are non-zero."""
    # Given
    agent = _make_agent(jax.random.PRNGKey(7))
    transitions = _make_transitions(jax.random.PRNGKey(2), n=MINIBATCH)

    features = jax.vmap(agent.actor)(transitions.obs)
    dists = jax.vmap(agent.action_head)(features)
    old_log_probs = jax.lax.stop_gradient(dists.log_prob(transitions.action.squeeze(-1)))
    advantages = jnp.ones(MINIBATCH)
    returns = jnp.ones(MINIBATCH)

    # When
    _loss, grads = eqx.filter_value_and_grad(lambda m: m._spo_loss(transitions, old_log_probs, advantages, returns))(
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
    """SPO satisfies the Agent protocol via structural subtyping."""
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
    agent = SPO(
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
    old_log_probs = jax.lax.stop_gradient(dists.log_prob(transitions.action.squeeze(-1)))
    advantages = jax.lax.stop_gradient(jnp.ones(MINIBATCH))
    returns = jax.lax.stop_gradient(jnp.ones(MINIBATCH))

    # When
    _loss, grads = eqx.filter_value_and_grad(lambda m: m._spo_loss(transitions, old_log_probs, advantages, returns))(
        agent
    )

    # Then
    critic_grad_leaves = jax.tree.leaves(eqx.filter(grads.critic, eqx.is_array))
    all_zero = all(jnp.allclose(g, 0.0) for g in critic_grad_leaves)
    assert all_zero, "Critic has non-zero gradients with beta_critic=0"


@pytest.mark.unit
def test_spo_loss_differs_from_ppo():
    """SPO and PPO losses should differ on the same data when the ratio
    deviates from 1, because they use different surrogate objectives."""
    # Given — build PPO and SPO with identical networks
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    shared_kwargs = dict(
        actor=MLP(OBS_DIM, HIDDEN, width=HIDDEN, depth=1, key=k1),
        action_head=DiscreteHead(HIDDEN, NUM_ACTIONS, key=k2),
        critic=MLP(OBS_DIM, 1, width=HIDDEN, depth=1, key=k3),
        optimizer=optax.adam(1e-3),
        gamma=0.99,
        tau=0.01,
        beta_critic=0.5,
        lambda_gae=0.95,
        eps_clip=0.2,
        num_epochs=1,
        minibatch_size=MINIBATCH,
    )
    ppo_agent = PPO(**shared_kwargs)
    spo_agent = SPO(**shared_kwargs)

    transitions = _make_transitions(jax.random.PRNGKey(1), n=MINIBATCH)

    # Compute current log probs then shift to create stale old_log_probs
    features = jax.vmap(ppo_agent.actor)(transitions.obs)
    dists = jax.vmap(ppo_agent.action_head)(features)
    current_lp = dists.log_prob(transitions.action.squeeze(-1))
    old_log_probs = current_lp - 1.5  # ratio ~ exp(1.5) ~ 4.5

    advantages = jnp.ones(MINIBATCH) * 2.0
    returns = jnp.ones(MINIBATCH)

    # When
    ppo_loss = ppo_agent._ppo_loss(transitions, old_log_probs, advantages, returns)
    spo_loss = spo_agent._spo_loss(transitions, old_log_probs, advantages, returns)

    # Then
    assert not jnp.allclose(ppo_loss, spo_loss, atol=1e-3), (
        f"SPO loss ({float(spo_loss):.4f}) equals PPO loss ({float(ppo_loss):.4f}) "
        "despite different surrogate objectives"
    )
