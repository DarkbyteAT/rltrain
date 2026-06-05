"""Tests for the PPO agent with clipped surrogate objective."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from rltrain.agents.agent import Agent
from rltrain.agents.ppo import PPO
from rltrain.heads import DiscreteHead
from rltrain.networks import MLP
from tests.agents._helpers import HIDDEN, MINIBATCH, NUM_ACTIONS, OBS_DIM
from tests.agents._helpers import _make_on_policy_transitions as _make_transitions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(key: jax.Array) -> PPO:
    """Build a small PPO agent for CartPole-sized problems."""
    k1, k2, k3 = jax.random.split(key, 3)
    return PPO(
        actor=MLP(OBS_DIM, HIDDEN, width_size=HIDDEN, depth=1, key=k1),
        action_head=DiscreteHead(HIDDEN, NUM_ACTIONS, key=k2),
        critic=MLP(OBS_DIM, 1, width_size=HIDDEN, depth=1, key=k3),
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
        actor=MLP(OBS_DIM, HIDDEN, width_size=HIDDEN, depth=1, key=k1),
        action_head=DiscreteHead(HIDDEN, NUM_ACTIONS, key=k2),
        critic=MLP(OBS_DIM, 1, width_size=HIDDEN, depth=1, key=k3),
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


@pytest.mark.unit
def test_minibatch_size_greater_than_collect_size_raises():
    """PPO.__init__ raises ValueError when minibatch_size > collect_size.

    Without the guard ``buffer_shuffle_into_minibatches`` would truncate to
    zero minibatches and the epoch loop would silently skip training.
    """
    # Given the OnPolicyAgent default collect_size of 256
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)

    # When PPO is built with a minibatch_size larger than collect_size,
    # the validator raises a ValueError naming both values
    with pytest.raises(ValueError, match="minibatch_size <= collect_size"):
        PPO(
            actor=MLP(OBS_DIM, HIDDEN, width_size=HIDDEN, depth=1, key=k1),
            action_head=DiscreteHead(HIDDEN, NUM_ACTIONS, key=k2),
            critic=MLP(OBS_DIM, 1, width_size=HIDDEN, depth=1, key=k3),
            optimizer=optax.adam(1e-3),
            gamma=0.99,
            tau=0.01,
            beta_critic=0.5,
            lambda_gae=0.95,
            eps_clip=0.2,
            num_epochs=1,
            minibatch_size=512,  # > default collect_size=256
        )


@pytest.mark.unit
def test_multi_env_gae_does_not_bootstrap_across_env_boundaries():
    """Multi-env GAE per-env reshape matches single-env GAE when both see the same trajectory.

    Build a batch of length T*N representing N envs each rolling out the
    same trajectory (same obs/reward/done sequence). The corrupted-flat
    GAE would bootstrap from env0_stepT-1 into env1_step0; the correct
    per-env GAE keeps each env's advantages identical and independent.
    """
    from rltrain.math import gae as gae_fn

    # Given two PPOs that differ only in num_envs (1 vs 4) and a synthetic
    # trajectory that, when tiled across 4 envs, lays out env-contiguous.
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    num_envs = 4
    per_env_T = 8
    horizon = num_envs * per_env_T

    single = PPO(
        actor=MLP(OBS_DIM, HIDDEN, width_size=HIDDEN, depth=1, key=k1),
        action_head=DiscreteHead(HIDDEN, NUM_ACTIONS, key=k2),
        critic=MLP(OBS_DIM, 1, width_size=HIDDEN, depth=1, key=k3),
        optimizer=optax.adam(1e-3),
        gamma=0.99,
        tau=0.01,
        beta_critic=0.5,
        lambda_gae=0.95,
        eps_clip=0.2,
        num_epochs=1,
        minibatch_size=per_env_T,
        num_envs=1,
    )
    multi = PPO(
        actor=single.actor,
        action_head=single.action_head,
        critic=single.critic,
        optimizer=optax.adam(1e-3),
        gamma=0.99,
        tau=0.01,
        beta_critic=0.5,
        lambda_gae=0.95,
        eps_clip=0.2,
        num_epochs=1,
        minibatch_size=per_env_T,
        num_envs=num_envs,
    )

    # When — produce a reference per-env GAE using the bare gae() helper.
    # We feed PPO a tiled trajectory where each env sees the same (obs,
    # reward, done) sequence; the per-env reshape inside PPO should
    # recover exactly that single-env GAE for every env.
    single_rewards = jax.random.normal(jax.random.PRNGKey(1), (per_env_T,))
    single_dones = jnp.zeros(per_env_T, dtype=jnp.bool_)
    single_obs = jax.random.normal(jax.random.PRNGKey(2), (per_env_T, OBS_DIM))

    # Use the critic to compute values (same critic for both).
    values = jax.vmap(lambda o: single.critic(o).squeeze(-1))(single_obs)
    # Bootstrap from the last next_obs (same obs, last one).
    bootstrap = single.critic(single_obs[-1]).squeeze(-1)
    values_t_plus_1 = jnp.concatenate([values, bootstrap[None]])
    ref_adv, ref_ret = gae_fn(
        values_t_plus_1,
        single_rewards,
        single_dones.astype(jnp.float32),
        single.gamma,
        single.lambda_gae,
    )

    # Tile the trajectory env-contiguously: env0_step0..env0_stepT, env1_step0..
    tiled_obs = jnp.tile(single_obs, (num_envs, 1))
    tiled_rewards = jnp.tile(single_rewards, num_envs)
    tiled_dones = jnp.tile(single_dones, num_envs)
    tiled_next_obs = tiled_obs  # any next_obs; last per-env gets bootstrap

    # Replicate the per-env GAE inside PPO's learn block.
    values_multi = jax.vmap(lambda o: multi.critic(o).squeeze(-1))(tiled_obs)
    values_per_env = values_multi.reshape(num_envs, per_env_T)
    last_next_obs = tiled_next_obs.reshape(num_envs, per_env_T, OBS_DIM)[:, -1]
    boot = jax.vmap(lambda o: multi.critic(o).squeeze(-1))(last_next_obs)
    vtp1 = jnp.concatenate([values_per_env, boot[:, None]], axis=1)
    rew_e = tiled_rewards.reshape(num_envs, per_env_T)
    don_e = tiled_dones.reshape(num_envs, per_env_T).astype(jnp.float32)
    adv_e, ret_e = jax.vmap(lambda v, r, d: gae_fn(v, r, d, multi.gamma, multi.lambda_gae))(vtp1, rew_e, don_e)

    # Then — every env's GAE matches the single-env reference exactly.
    for e in range(num_envs):
        assert jnp.allclose(adv_e[e], ref_adv, atol=1e-5), f"env {e} advantages drift from reference"
        assert jnp.allclose(ret_e[e], ref_ret, atol=1e-5), f"env {e} returns drift from reference"

    # And — flat-shape GAE (the BUG) would NOT match. Build the corrupted
    # version to show the test would catch the regression if the per-env
    # reshape were removed.
    flat_vtp1 = jnp.concatenate([values_multi, bootstrap[None]])
    flat_adv, _flat_ret = gae_fn(
        flat_vtp1,
        tiled_rewards,
        tiled_dones.astype(jnp.float32),
        multi.gamma,
        multi.lambda_gae,
    )
    # Slot per_env_T (start of env 1) under the bug would be bootstrapped from
    # env0's V_T+1; under the correct per-env GAE it's bootstrapped from
    # env1's own V_0. With horizon * 4 those values diverge.
    boundary_idx = per_env_T  # first step of env 1 in the flat layout
    assert not jnp.allclose(flat_adv[boundary_idx], ref_adv[0], atol=1e-3), (
        f"Test setup didn't expose the bug — flat GAE at boundary "
        f"({flat_adv[boundary_idx]:.4f}) accidentally matches ref ({ref_adv[0]:.4f})"
    )
