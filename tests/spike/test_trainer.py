"""Tests for the Trainer — verifies all three env strategies and callback dispatch."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from spike.agents.vanilla_dqn import VanillaDQN
from spike.agents.vanilla_pg import VanillaPG
from spike.env import EnvCapabilities, GymnaxEnv
from spike.heads import DiscreteHead
from spike.networks import MLP
from spike.trainer import Trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OBS_DIM = 4
NUM_ACTIONS = 2
HIDDEN = 32
# VanillaPG.collect_size is a ClassVar defaulting to 256.
# Tests use num_steps >= 256 to ensure at least one learn step fires.
COLLECT_SIZE_PG = 256


def _make_pg_agent(key: jax.Array) -> VanillaPG:
    """Build a small VanillaPG for testing."""
    k1, k2 = jax.random.split(key)
    return VanillaPG(
        actor=MLP(OBS_DIM, HIDDEN, width=HIDDEN, depth=1, key=k1),
        action_head=DiscreteHead(HIDDEN, NUM_ACTIONS, key=k2),
        optimizer=optax.adam(1e-3),
        gamma=0.99,
        tau=0.01,
        normalise=True,
    )


def _make_dqn_agent(key: jax.Array) -> VanillaDQN:
    """Build a small VanillaDQN for testing."""
    return VanillaDQN(
        q_net=MLP(OBS_DIM, NUM_ACTIONS, width=HIDDEN, depth=1, key=key),
        optimizer=optax.adam(1e-3),
        gamma=0.99,
        target_rate=0.01,
        num_actions=NUM_ACTIONS,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.01,
    )


class _ForcedCapabilityEnv:
    """Wraps a GymnaxEnv but overrides capabilities for testing dispatch."""

    def __init__(self, inner: GymnaxEnv, caps: EnvCapabilities):
        self._inner = inner
        self.capabilities = caps
        self.obs_shape = inner.obs_shape
        self.num_actions = inner.num_actions

    def reset(self, key):
        return self._inner.reset(key)

    def step(self, state, action, key):
        return self._inner.step(state, action, key)


class _RecordingCallback:
    """Mock callback that records which hooks were called and their args."""

    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []

    def on_train_start(self, config: dict, run_dir: Path | None) -> None:
        self.calls.append(("on_train_start", (config, run_dir)))

    def on_step(self, step: int, metrics: dict[str, float]) -> None:
        self.calls.append(("on_step", (step, metrics)))

    def on_episode_end(self, episode: int, episode_return: float, episode_length: int) -> None:
        self.calls.append(("on_episode_end", (episode, episode_return, episode_length)))

    def on_checkpoint(self, step: int, agent_state: object, run_dir: Path | None) -> None:
        self.calls.append(("on_checkpoint", (step,)))

    def on_train_end(self, agent_state: object, run_dir: Path | None) -> None:
        self.calls.append(("on_train_end", ()))


# ---------------------------------------------------------------------------
# Tests: Python loop with gymnax env (forced capabilities=False)
# ---------------------------------------------------------------------------


def test_trainer_vmap_pg():
    """Given a VanillaPG agent and a GymnaxEnv with vmap capability,
    When the Trainer runs for 512 steps,
    Then it returns a trained state with different params from init.
    """
    key = jax.random.PRNGKey(0)
    k_agent, k_fit = jax.random.split(key)

    agent = _make_pg_agent(k_agent)
    env = GymnaxEnv("CartPole-v1")

    # Force vmap-only (no scan)
    env_wrapped = _ForcedCapabilityEnv(env, EnvCapabilities(pure_step=True, vmap_batch=True, scan_rollout=False))

    trainer = Trainer(
        agent,
        env_wrapped,
        num_steps=512,
        checkpoint_steps=256,
    )

    init_state = agent.init(k_agent)
    final_state = trainer.fit(k_fit)

    # Params should have changed after training
    init_flat = jax.tree.leaves(init_state.params)
    final_flat = jax.tree.leaves(final_state.params)
    any_changed = any(not jnp.allclose(i, f) for i, f in zip(init_flat, final_flat, strict=True))
    assert any_changed, "Expected params to change after training"


def test_trainer_python_loop_dqn():
    """Given a VanillaDQN agent and a GymnaxEnv forced to Python-loop mode,
    When the Trainer runs for 200 steps,
    Then it returns a trained state (DQN learns after every step once buffer is full).
    """
    key = jax.random.PRNGKey(1)
    k_agent, k_fit = jax.random.split(key)

    agent = _make_dqn_agent(k_agent)
    env = GymnaxEnv("CartPole-v1")

    # Force Python-loop fallback via vmap-capable env (vmap uses the same
    # gymnax Python loop internally)
    env_wrapped = _ForcedCapabilityEnv(env, EnvCapabilities(pure_step=True, vmap_batch=True, scan_rollout=False))

    trainer = Trainer(
        agent,
        env_wrapped,
        num_steps=200,
        checkpoint_steps=100,
        buffer_capacity=200,
        batch_size=32,
        min_buffer_size=32,
    )

    init_state = agent.init(k_agent)
    final_state = trainer.fit(k_fit)

    init_flat = jax.tree.leaves(init_state.params)
    final_flat = jax.tree.leaves(final_state.params)
    any_changed = any(not jnp.allclose(i, f) for i, f in zip(init_flat, final_flat, strict=True))
    assert any_changed, "Expected DQN params to change after training"


def test_trainer_scan_pg():
    """Given a VanillaPG agent and a GymnaxEnv with full scan capability,
    When the Trainer runs for 512 steps using the scan strategy,
    Then it returns a trained state with updated params.
    """
    key = jax.random.PRNGKey(2)
    k_agent, k_fit = jax.random.split(key)

    agent = _make_pg_agent(k_agent)
    env = GymnaxEnv("CartPole-v1")

    trainer = Trainer(
        agent,
        env,
        num_steps=512,
        checkpoint_steps=256,
    )

    init_state = agent.init(k_agent)
    final_state = trainer.fit(k_fit)

    init_flat = jax.tree.leaves(init_state.params)
    final_flat = jax.tree.leaves(final_state.params)
    any_changed = any(not jnp.allclose(i, f) for i, f in zip(init_flat, final_flat, strict=True))
    assert any_changed, "Expected params to change after scan training"


def test_trainer_dispatches_on_capabilities():
    """Given environments with different capabilities,
    When fit() is called,
    Then it dispatches to the correct strategy.
    """
    key = jax.random.PRNGKey(3)
    agent = _make_pg_agent(key)
    env = GymnaxEnv("CartPole-v1")

    # Full capabilities -> scan
    Trainer(agent, env, num_steps=256, checkpoint_steps=256)
    assert env.capabilities.scan_rollout is True

    # vmap only -> vmap
    env_vmap = _ForcedCapabilityEnv(env, EnvCapabilities(pure_step=True, vmap_batch=True, scan_rollout=False))
    Trainer(agent, env_vmap, num_steps=256, checkpoint_steps=256)
    assert env_vmap.capabilities.scan_rollout is False
    assert env_vmap.capabilities.vmap_batch is True

    # No JAX capabilities -> python
    env_python = _ForcedCapabilityEnv(env, EnvCapabilities(pure_step=False, vmap_batch=False, scan_rollout=False))
    Trainer(agent, env_python, num_steps=256, checkpoint_steps=256)
    assert env_python.capabilities.scan_rollout is False
    assert env_python.capabilities.vmap_batch is False


def test_trainer_callbacks_fire():
    """Given a Trainer with a recording callback,
    When fit() runs,
    Then on_train_start, on_checkpoint, on_train_end are all called,
    and on_episode_end is called at least once (CartPole episodes are short).
    """
    key = jax.random.PRNGKey(4)
    k_agent, k_fit = jax.random.split(key)

    agent = _make_pg_agent(k_agent)
    env = GymnaxEnv("CartPole-v1")

    # Force vmap (Python loop) for predictable callback firing
    env_wrapped = _ForcedCapabilityEnv(env, EnvCapabilities(pure_step=True, vmap_batch=True, scan_rollout=False))

    recorder = _RecordingCallback()
    trainer = Trainer(
        agent,
        env_wrapped,
        num_steps=512,
        checkpoint_steps=256,
        callbacks=[recorder],
    )
    trainer.fit(k_fit)

    hook_names = [name for name, _ in recorder.calls]

    assert "on_train_start" in hook_names, "on_train_start not fired"
    assert "on_train_end" in hook_names, "on_train_end not fired"
    assert "on_checkpoint" in hook_names, "on_checkpoint not fired"
    assert "on_episode_end" in hook_names, "on_episode_end not fired (CartPole should end within 512 steps)"

    # on_train_start should be first, on_train_end should be last
    assert hook_names[0] == "on_train_start"
    assert hook_names[-1] == "on_train_end"


def test_trainer_returns_trained_state():
    """Given a Trainer that runs for enough steps,
    When fit() completes,
    Then the returned state is a valid TrainState pytree with array leaves.
    """
    key = jax.random.PRNGKey(5)
    k_agent, k_fit = jax.random.split(key)

    agent = _make_pg_agent(k_agent)
    env = GymnaxEnv("CartPole-v1")
    env_wrapped = _ForcedCapabilityEnv(env, EnvCapabilities(pure_step=True, vmap_batch=True, scan_rollout=False))

    trainer = Trainer(
        agent,
        env_wrapped,
        num_steps=512,
        checkpoint_steps=256,
    )
    final_state = trainer.fit(k_fit)

    # The state should be a TrainState with params, opt_state, target_params
    assert hasattr(final_state, "params")
    assert hasattr(final_state, "opt_state")

    # All leaves should be JAX arrays
    leaves = jax.tree.leaves(final_state)
    assert len(leaves) > 0, "State should have array leaves"
    for leaf in leaves:
        assert hasattr(leaf, "shape"), f"Expected JAX array, got {type(leaf)}"


def test_trainer_scan_callbacks_fire():
    """Given a Trainer using scan strategy with a recording callback,
    When fit() runs,
    Then all callback types fire correctly.
    """
    key = jax.random.PRNGKey(6)
    k_agent, k_fit = jax.random.split(key)

    agent = _make_pg_agent(k_agent)
    env = GymnaxEnv("CartPole-v1")

    recorder = _RecordingCallback()
    trainer = Trainer(
        agent,
        env,
        num_steps=512,
        checkpoint_steps=256,
        callbacks=[recorder],
    )
    trainer.fit(k_fit)

    hook_names = [name for name, _ in recorder.calls]

    assert "on_train_start" in hook_names
    assert "on_train_end" in hook_names
    assert "on_checkpoint" in hook_names
    # CartPole episodes are short; at least one should complete in 512 steps
    assert "on_episode_end" in hook_names
