"""Tests for the pure-functional VanillaPG agent."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from rltrain.agents.vanilla_pg import VanillaPG
from rltrain.env import GymnaxEnv
from rltrain.heads import DiscreteHead
from rltrain.networks import MLP
from rltrain.transitions import Transition, make_transition
from tests.agents._helpers import HIDDEN, NUM_ACTIONS, OBS_DIM
from tests.agents._helpers import _make_on_policy_transitions as _make_transitions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(key: jax.Array) -> VanillaPG:
    """Build a small VanillaPG for CartPole-sized problems."""
    k1, k2 = jax.random.split(key)
    return VanillaPG(
        actor=MLP(OBS_DIM, HIDDEN, width=HIDDEN, depth=1, key=k1),
        action_head=DiscreteHead(HIDDEN, NUM_ACTIONS, key=k2),
        optimizer=optax.adam(1e-3),
        gamma=0.99,
        tau=0.01,
        normalise=True,
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_act_returns_valid_action():
    """Given an observation, act produces an action in valid range."""
    # Given
    key = jax.random.PRNGKey(0)
    agent = _make_agent(key)
    state = agent.init(jax.random.PRNGKey(1))
    obs = jnp.ones(OBS_DIM)

    # When
    action = agent.act(state, obs, jax.random.PRNGKey(2))

    # Then
    assert action.shape == ()
    assert 0 <= int(action) < NUM_ACTIONS


@pytest.mark.unit
def test_loss_is_scalar():
    """Given a batch of transitions, loss returns a finite scalar."""
    # Given
    key = jax.random.PRNGKey(42)
    agent = _make_agent(key)
    transitions = _make_transitions(jax.random.PRNGKey(1))

    # When
    loss_val = agent._loss(transitions)

    # Then
    assert loss_val.shape == ()
    assert jnp.isfinite(loss_val)


@pytest.mark.unit
def test_gradients_flow():
    """Gradients through loss are non-zero — the loss depends on parameters."""
    # Given
    key = jax.random.PRNGKey(7)
    agent = _make_agent(key)
    transitions = _make_transitions(jax.random.PRNGKey(2))

    # When
    _loss, grads = eqx.filter_value_and_grad(lambda m: m._loss(transitions))(agent)

    # Then — at least some gradient leaves are non-zero
    grad_leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
    has_nonzero = any(jnp.any(g != 0.0) for g in grad_leaves)
    assert has_nonzero, "All gradients are zero — loss is disconnected from parameters"


@pytest.mark.unit
def test_learn_updates_params():
    """After one learn step, at least some parameters differ from the originals."""
    # Given
    key = jax.random.PRNGKey(99)
    agent = _make_agent(key)
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


# ---------------------------------------------------------------------------
# End-to-end training test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_trains_cartpole():
    """Train VanillaPG on CartPole-v1 (gymnax) for ~50K steps.

    Success criterion: mean return over the last 20 episodes exceeds 50.
    This is not perfect play (max 500) — just evidence of a learning signal.
    """
    # Given
    env = GymnaxEnv("CartPole-v1")
    key = jax.random.PRNGKey(0)
    k_agent, k_env, k_loop = jax.random.split(key, 3)

    agent = VanillaPG(
        actor=MLP(env.obs_shape[0], 64, width=64, depth=1, key=jax.random.split(k_agent)[0]),
        action_head=DiscreteHead(64, env.num_actions, key=jax.random.split(k_agent)[1]),
        optimizer=optax.adam(3e-3),
        gamma=0.99,
        tau=0.01,
        normalise=True,
    )

    state = agent.init(jax.random.PRNGKey(42))
    env_state = env.reset(k_env)

    episode_returns: list[float] = []
    total_steps = 0
    max_steps = 50_000

    # When — Python collection loop, jitted learn
    jit_learn = eqx.filter_jit(agent.learn)

    while total_steps < max_steps:
        k_loop, k_ep = jax.random.split(k_loop)
        ep_transitions: list[Transition] = []
        ep_return = 0.0

        while True:
            k_ep, k_act, k_step = jax.random.split(k_ep, 3)

            action = agent.act(state, env_state.obs, k_act)

            prev_obs = env_state.obs
            env_state = env.step(env_state, action, k_step)

            ep_transitions.append(
                make_transition(
                    obs=prev_obs,
                    action=action,
                    reward=env_state.reward,
                    next_obs=env_state.obs,
                    done=env_state.done,
                )
            )
            ep_return += float(env_state.reward)
            total_steps += 1

            if bool(env_state.done) or len(ep_transitions) >= 500:
                break

        episode_returns.append(ep_return)

        # Stack individual transitions into a batched Transition
        transitions = jax.tree.map(lambda *xs: jnp.stack(xs), *ep_transitions)
        state, _metrics = jit_learn(state, transitions, jax.random.PRNGKey(0))

    # Then
    recent = episode_returns[-20:]
    mean_return = sum(recent) / len(recent)
    assert mean_return > 50, (
        f"Mean return over last 20 episodes was {mean_return:.1f}, expected > 50. "
        f"Full return history (last 40): {episode_returns[-40:]}"
    )
