"""Tests for the Agent Protocol's ``act_batch`` default implementation.

The Protocol default vmaps ``act`` over a batched leading axis. These
tests exercise the contract on representative agents from each family
(on-policy via PPO, off-policy via VanillaDQN) — the default is shared,
so passing on one of each implies the default works uniformly.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from rltrain.agents.ppo import PPO
from rltrain.agents.vanilla_dqn import VanillaDQN
from rltrain.heads import DiscreteHead
from rltrain.networks import MLP
from tests.agents._helpers import HIDDEN, NUM_ACTIONS, OBS_DIM


def _make_ppo(key: jax.Array) -> PPO:
    """Build a small PPO for batched-act probing."""
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
        num_epochs=1,
        minibatch_size=8,
    )


def _make_dqn(key: jax.Array) -> VanillaDQN:
    """Build a small VanillaDQN for batched-act probing."""
    return VanillaDQN(
        q_net=MLP(OBS_DIM, NUM_ACTIONS, width=HIDDEN, depth=1, key=key),
        optimizer=optax.adam(1e-3),
        gamma=0.99,
        target_rate=0.005,
        num_actions=NUM_ACTIONS,
        eps_start=0.1,
        eps_end=0.01,
        eps_decay=1e-4,
    )


@pytest.mark.unit
def test_ppo_act_batch_shape() -> None:
    """PPO.act_batch maps obs of shape (N, obs_dim) to actions of shape (N,)."""
    # Given a PPO and a batch of N observations
    key = jax.random.key(0)
    k_agent, k_state, k_act = jax.random.split(key, 3)
    agent = _make_ppo(k_agent)
    state = agent.init(k_state)
    obs_batch = jax.random.normal(k_act, (5, OBS_DIM))

    # When act_batch is called
    actions = agent.act_batch(state, obs_batch, k_act)

    # Then the returned shape matches the input batch dimension
    assert actions.shape == (5,)


@pytest.mark.unit
def test_dqn_act_batch_shape() -> None:
    """VanillaDQN.act_batch handles batched obs uniformly with the default vmap."""
    # Given a DQN and a batch of N observations
    key = jax.random.key(1)
    k_agent, k_state, k_act = jax.random.split(key, 3)
    agent = _make_dqn(k_agent)
    state = agent.init(k_state)
    obs_batch = jax.random.normal(k_act, (7, OBS_DIM))

    # When act_batch is called
    actions = agent.act_batch(state, obs_batch, k_act)

    # Then we get one action per env
    assert actions.shape == (7,)


@pytest.mark.unit
def test_act_batch_is_jit_compatible() -> None:
    """act_batch composes with eqx.filter_jit without recompiling on call."""
    # Given a PPO, a state, and a jitted act_batch
    key = jax.random.key(2)
    k_agent, k_state, k_act = jax.random.split(key, 3)
    agent = _make_ppo(k_agent)
    state = agent.init(k_state)
    jitted = eqx.filter_jit(agent.act_batch)

    # When the jitted function is called twice
    obs_batch = jnp.zeros((4, OBS_DIM))
    a1 = jitted(state, obs_batch, k_act)
    a2 = jitted(state, obs_batch, k_act)

    # Then both calls produce same-shape outputs and identical actions
    # (deterministic given fixed key + state)
    assert a1.shape == (4,)
    assert jnp.array_equal(a1, a2)


@pytest.mark.unit
def test_act_batch_matches_per_element_act() -> None:
    """Default vmap'd act_batch produces the same actions as N sequential act calls."""
    # Given a PPO and a batch of distinct observations
    key = jax.random.key(3)
    k_agent, k_state, k_obs, k_act = jax.random.split(key, 4)
    agent = _make_ppo(k_agent)
    state = agent.init(k_state)
    obs_batch = jax.random.normal(k_obs, (3, OBS_DIM))

    # When act_batch is invoked with key k_act
    actions_batch = agent.act_batch(state, obs_batch, k_act)

    # Then per-element act calls with split sub-keys produce the same actions
    sub_keys = jax.random.split(k_act, 3)
    actions_seq = jnp.stack([agent.act(state, obs_batch[i], sub_keys[i]) for i in range(3)])
    assert jnp.array_equal(actions_batch, actions_seq)
