"""End-to-end smoke tests — verify each agent trains end-to-end on a
single env without crashing and produces a finite final state.

These tests deliberately use tiny step counts (~500-1000 per agent) so
the full suite finishes in seconds. They prove the training pipeline
wires together — agent.init → trainer.fit → final TrainState exists
with finite leaves. Convergence claims live in
``tests/test_e2e_training_slow.py`` (marked ``@pytest.mark.slow``).
"""

import jax
import jax.numpy as jnp
import optax
import pytest

from rltrain.agents.distributional_dqn import DistributionalDQN
from rltrain.agents.double_dqn import DoubleDQN
from rltrain.agents.ppo import PPO
from rltrain.agents.sac import SAC
from rltrain.agents.spo import SPO
from rltrain.agents.vanilla_dqn import VanillaDQN
from rltrain.env import GymnaxEnv
from rltrain.heads import CategoricalAtomHead, DiscreteHead, GaussianHead, SquashedGaussianHead
from rltrain.networks import MLP
from rltrain.trainer import Trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


CARTPOLE_OBS = 4
CARTPOLE_ACTS = 2
PENDULUM_OBS = 3
PENDULUM_ACTS = 1

# Smoke configuration: just enough to exercise wiring; not enough to converge.
SMOKE_STEPS = 1000
SMOKE_CHECKPOINT = 250


def _assert_state_finite(state) -> None:
    """Recurse the state pytree and assert every array leaf is finite."""
    leaves = jax.tree_util.tree_leaves(state)
    for leaf in leaves:
        if hasattr(leaf, "shape"):
            assert jnp.all(jnp.isfinite(leaf)), f"non-finite leaf in state: shape={leaf.shape}"


# ---------------------------------------------------------------------------
# Discrete-action agents on CartPole-v1
# ---------------------------------------------------------------------------


@pytest.mark.e2e
def test_ppo_cartpole():
    """PPO completes one training segment on CartPole and produces a finite state."""
    # Given
    key = jax.random.PRNGKey(0)
    k1, k2, k3, key = jax.random.split(key, 4)
    agent = PPO(
        actor=MLP(CARTPOLE_OBS, 64, width=64, depth=1, key=k1),
        action_head=DiscreteHead(64, CARTPOLE_ACTS, key=k2),
        critic=MLP(CARTPOLE_OBS, 1, width=64, depth=1, key=k3),
        optimizer=optax.adam(3e-3),
        gamma=0.99,
        tau=0.01,
        beta_critic=0.5,
        lambda_gae=0.95,
        eps_clip=0.2,
        num_epochs=4,
        minibatch_size=64,
    )
    env = GymnaxEnv("CartPole-v1")
    trainer = Trainer(agent, env, num_steps=SMOKE_STEPS, checkpoint_steps=SMOKE_CHECKPOINT)

    # When
    state = trainer.fit(key)

    # Then
    _assert_state_finite(state)


@pytest.mark.e2e
def test_spo_cartpole():
    """SPO completes one training segment on CartPole and produces a finite state."""
    key = jax.random.PRNGKey(1)
    k1, k2, k3, key = jax.random.split(key, 4)
    agent = SPO(
        actor=MLP(CARTPOLE_OBS, 64, width=64, depth=1, key=k1),
        action_head=DiscreteHead(64, CARTPOLE_ACTS, key=k2),
        critic=MLP(CARTPOLE_OBS, 1, width=64, depth=1, key=k3),
        optimizer=optax.adam(3e-3),
        gamma=0.99,
        tau=0.01,
        beta_critic=0.5,
        lambda_gae=0.95,
        eps_clip=0.2,
        num_epochs=4,
        minibatch_size=64,
    )
    env = GymnaxEnv("CartPole-v1")
    trainer = Trainer(agent, env, num_steps=SMOKE_STEPS, checkpoint_steps=SMOKE_CHECKPOINT)
    state = trainer.fit(key)
    _assert_state_finite(state)


@pytest.mark.e2e
def test_vanilla_dqn_cartpole():
    """VanillaDQN completes one training segment on CartPole and produces a finite state."""
    key = jax.random.PRNGKey(2)
    k1, key = jax.random.split(key)
    agent = VanillaDQN(
        q_net=MLP(CARTPOLE_OBS, CARTPOLE_ACTS, width=128, depth=2, key=k1),
        optimizer=optax.adam(3e-4),
        gamma=0.99,
        target_rate=0.005,
        num_actions=CARTPOLE_ACTS,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=5e-4,
    )
    env = GymnaxEnv("CartPole-v1")
    trainer = Trainer(
        agent,
        env,
        num_steps=SMOKE_STEPS,
        checkpoint_steps=SMOKE_CHECKPOINT,
        buffer_capacity=10_000,
        batch_size=64,
    )
    state = trainer.fit(key)
    _assert_state_finite(state)


@pytest.mark.e2e
def test_double_dqn_cartpole():
    """DoubleDQN completes one training segment on CartPole and produces a finite state."""
    key = jax.random.PRNGKey(3)
    k1, key = jax.random.split(key)
    agent = DoubleDQN(
        q_net=MLP(CARTPOLE_OBS, CARTPOLE_ACTS, width=128, depth=2, key=k1),
        optimizer=optax.adam(3e-4),
        gamma=0.99,
        target_rate=0.005,
        num_actions=CARTPOLE_ACTS,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=5e-4,
    )
    env = GymnaxEnv("CartPole-v1")
    trainer = Trainer(
        agent,
        env,
        num_steps=SMOKE_STEPS,
        checkpoint_steps=SMOKE_CHECKPOINT,
        buffer_capacity=10_000,
        batch_size=64,
    )
    state = trainer.fit(key)
    _assert_state_finite(state)


@pytest.mark.e2e
def test_c51_cartpole():
    """DistributionalDQN (C51) completes one training segment on CartPole and produces a finite state."""
    key = jax.random.PRNGKey(4)
    k1, k2, key = jax.random.split(key, 3)
    agent = DistributionalDQN(
        feature_net=MLP(CARTPOLE_OBS, 64, width=64, depth=1, key=k1),
        atom_head=CategoricalAtomHead(64, CARTPOLE_ACTS, num_atoms=21, v_min=-10, v_max=10, key=k2),
        optimizer=optax.adam(3e-4),
        gamma=0.99,
        target_rate=0.005,
        num_actions=CARTPOLE_ACTS,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=5e-4,
    )
    env = GymnaxEnv("CartPole-v1")
    trainer = Trainer(
        agent,
        env,
        num_steps=SMOKE_STEPS,
        checkpoint_steps=SMOKE_CHECKPOINT,
        buffer_capacity=10_000,
        batch_size=64,
    )
    state = trainer.fit(key)
    _assert_state_finite(state)


@pytest.mark.e2e
def test_sac_discrete_cartpole():
    """SAC-Discrete completes one training segment on CartPole and produces a finite state."""
    key = jax.random.PRNGKey(5)
    k1, k2, k3, k4, key = jax.random.split(key, 5)
    agent = SAC(
        actor=MLP(CARTPOLE_OBS, 64, width=64, depth=1, key=k1),
        action_head=DiscreteHead(64, CARTPOLE_ACTS, key=k2),
        critic_1=MLP(CARTPOLE_OBS, CARTPOLE_ACTS, width=64, depth=1, key=k3),
        critic_2=MLP(CARTPOLE_OBS, CARTPOLE_ACTS, width=64, depth=1, key=k4),
        actor_optimizer=optax.adam(3e-4),
        critic_optimizer=optax.adam(3e-4),
        alpha_optimizer=optax.adam(3e-4),
        gamma=0.99,
        tau=0.005,
    )
    env = GymnaxEnv("CartPole-v1")
    trainer = Trainer(
        agent,
        env,
        num_steps=SMOKE_STEPS,
        checkpoint_steps=SMOKE_CHECKPOINT,
        buffer_capacity=10_000,
        batch_size=64,
    )
    state = trainer.fit(key)
    _assert_state_finite(state)


# ---------------------------------------------------------------------------
# Continuous-action agents on Pendulum-v1
# ---------------------------------------------------------------------------


@pytest.mark.e2e
def test_ppo_continuous_pendulum():
    """PPO with GaussianHead completes one training segment on Pendulum."""
    key = jax.random.PRNGKey(6)
    k1, k2, k3, key = jax.random.split(key, 4)
    agent = PPO(
        actor=MLP(PENDULUM_OBS, 64, width=64, depth=1, key=k1),
        action_head=GaussianHead(64, PENDULUM_ACTS, key=k2),
        critic=MLP(PENDULUM_OBS, 1, width=64, depth=1, key=k3),
        optimizer=optax.adam(3e-3),
        gamma=0.99,
        tau=0.01,
        beta_critic=0.5,
        lambda_gae=0.95,
        eps_clip=0.2,
        num_epochs=4,
        minibatch_size=64,
    )
    env = GymnaxEnv("Pendulum-v1")
    trainer = Trainer(agent, env, num_steps=SMOKE_STEPS, checkpoint_steps=SMOKE_CHECKPOINT)
    state = trainer.fit(key)
    _assert_state_finite(state)


@pytest.mark.e2e
def test_sac_continuous_pendulum():
    """SAC with SquashedGaussianHead completes one training segment on Pendulum."""
    key = jax.random.PRNGKey(7)
    k1, k2, k3, k4, key = jax.random.split(key, 5)
    obs_act = PENDULUM_OBS + PENDULUM_ACTS
    agent = SAC(
        actor=MLP(PENDULUM_OBS, 64, width=64, depth=1, key=k1),
        action_head=SquashedGaussianHead(64, PENDULUM_ACTS, key=k2),
        critic_1=MLP(obs_act, 1, width=64, depth=1, key=k3),
        critic_2=MLP(obs_act, 1, width=64, depth=1, key=k4),
        actor_optimizer=optax.adam(3e-4),
        critic_optimizer=optax.adam(3e-4),
        alpha_optimizer=optax.adam(3e-4),
        gamma=0.99,
        tau=0.005,
    )
    env = GymnaxEnv("Pendulum-v1")
    trainer = Trainer(
        agent,
        env,
        num_steps=SMOKE_STEPS,
        checkpoint_steps=SMOKE_CHECKPOINT,
        buffer_capacity=10_000,
        batch_size=64,
    )
    state = trainer.fit(key)
    _assert_state_finite(state)
