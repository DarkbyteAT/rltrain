"""Tests for the VideoRecorderCallback."""

from unittest.mock import MagicMock

import gymnasium as gym
import gymnasium.vector as vgym
import numpy as np

from rltrain.callbacks.video_recorder import VideoRecorderCallback
from rltrain.env import MDP


def _make_mdp() -> MDP:
    """Build a minimal MDP wrapping CartPole."""
    env = vgym.SyncVectorEnv([lambda: gym.make("CartPole-v1")])
    return MDP(env, run_beta=0.1, log_freq=100, swap_channels=False)


def _make_agent_stub() -> MagicMock:
    """Create a stub agent whose __call__ returns a valid CartPole action."""
    agent = MagicMock()
    agent.side_effect = lambda obs: np.array([0])
    return agent


def test_auto_detect_creates_eval_env(tmp_path):
    """With no env_fn, on_train_start should auto-detect from MDP spec."""
    mdp = _make_mdp()
    mdp.setup(seed=42)
    cb = VideoRecorderCallback()
    cb.on_train_start(_make_agent_stub(), mdp, tmp_path)

    assert cb._eval_env is not None
    assert cb._enabled is True
    assert (tmp_path / "videos").is_dir()


def test_explicit_env_fn_used(tmp_path):
    """When env_fn is provided, it should be used instead of auto-detection."""
    factory_called = False

    def my_factory():
        nonlocal factory_called
        factory_called = True
        return gym.make("CartPole-v1", render_mode="rgb_array")

    mdp = _make_mdp()
    mdp.setup(seed=42)
    cb = VideoRecorderCallback(env_fn=my_factory)
    cb.on_train_start(_make_agent_stub(), mdp, tmp_path)

    assert factory_called
    assert cb._eval_env is not None


def test_graceful_disable_on_non_renderable_env(tmp_path):
    """If env can't render rgb_array, callback disables itself."""

    def bad_factory():
        return gym.make("CartPole-v1")  # no render_mode

    mdp = _make_mdp()
    mdp.setup(seed=42)
    cb = VideoRecorderCallback(env_fn=bad_factory)
    cb.on_train_start(_make_agent_stub(), mdp, tmp_path)

    assert cb._enabled is False
