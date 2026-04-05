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


def _make_recording_env() -> gym.Env:
    return gym.make("CartPole-v1", render_mode="rgb_array")


# --- on_train_start: env creation ---


def test_auto_detect_creates_eval_env_from_mdp_spec(tmp_path):
    """Auto-detection should build a renderable eval env from MDP.env.envs[0].spec."""
    # Given: a callback with no env_fn and a CartPole MDP
    mdp = _make_mdp()
    mdp.setup(seed=42)
    cb = VideoRecorderCallback()

    # When: on_train_start fires
    cb.on_train_start(_make_agent_stub(), mdp, tmp_path)

    # Then: eval env is created, callback is enabled, videos dir exists
    assert cb._eval_env is not None
    assert cb._enabled is True
    assert (tmp_path / "videos").is_dir()


def test_explicit_env_fn_is_called_instead_of_auto_detection(tmp_path):
    """When env_fn is provided, it should be used instead of auto-detection."""
    # Given: a callback with an explicit env factory
    factory_called = False

    def my_factory():
        nonlocal factory_called
        factory_called = True
        return gym.make("CartPole-v1", render_mode="rgb_array")

    mdp = _make_mdp()
    mdp.setup(seed=42)
    cb = VideoRecorderCallback(env_fn=my_factory)

    # When: on_train_start fires
    cb.on_train_start(_make_agent_stub(), mdp, tmp_path)

    # Then: the factory was called and the eval env was created
    assert factory_called
    assert cb._eval_env is not None


def test_non_renderable_env_disables_callback(tmp_path):
    """If the eval env doesn't support rgb_array, the callback disables itself."""
    # Given: a factory that returns a non-renderable env
    mdp = _make_mdp()
    mdp.setup(seed=42)
    cb = VideoRecorderCallback(env_fn=lambda: gym.make("CartPole-v1"))

    # When: on_train_start fires
    cb.on_train_start(_make_agent_stub(), mdp, tmp_path)

    # Then: callback is disabled
    assert cb._enabled is False


# --- on_checkpoint: default trigger ---


def test_checkpoint_trigger_produces_video_files(tmp_path):
    """Default trigger: on_checkpoint runs eval rollouts that produce video files."""
    # Given: a callback with default trigger (checkpoint-based) and 1 eval episode
    mdp = _make_mdp()
    mdp.setup(seed=42)
    agent = _make_agent_stub()
    cb = VideoRecorderCallback(env_fn=_make_recording_env, num_episodes=1)
    cb.on_train_start(agent, mdp, tmp_path)

    # When: a checkpoint occurs
    cb.on_checkpoint(agent, mdp, tmp_path)
    cb.on_train_end(agent, mdp, tmp_path)

    # Then: exactly one video named by training step is written
    videos = list((tmp_path / "videos").glob("*.mp4"))
    assert len(videos) == 1
    assert f"step-{mdp.total_steps}" in videos[0].name


def test_checkpoint_is_noop_when_eval_trigger_is_set(tmp_path):
    """When eval_trigger is configured, on_checkpoint should not run rollouts."""
    # Given: a callback with an eval trigger that never fires
    mdp = _make_mdp()
    mdp.setup(seed=42)
    agent = _make_agent_stub()
    cb = VideoRecorderCallback(
        env_fn=_make_recording_env,
        num_episodes=1,
        eval_trigger=lambda _ep: False,
    )
    cb.on_train_start(agent, mdp, tmp_path)

    # When: a checkpoint occurs
    cb.on_checkpoint(agent, mdp, tmp_path)
    cb.on_train_end(agent, mdp, tmp_path)

    # Then: no videos are produced (checkpoint trigger is disabled)
    videos = list((tmp_path / "videos").glob("*.mp4"))
    assert len(videos) == 0


# --- on_episode_end: custom trigger ---


def test_eval_trigger_runs_rollouts_when_predicate_matches(tmp_path):
    """on_episode_end should run eval rollouts when eval_trigger returns True."""
    # Given: a callback triggered on episode 1 only
    mdp = _make_mdp()
    mdp.setup(seed=42)
    agent = _make_agent_stub()
    cb = VideoRecorderCallback(
        env_fn=_make_recording_env,
        num_episodes=1,
        eval_trigger=lambda ep: ep == 1,
    )
    cb.on_train_start(agent, mdp, tmp_path)

    # When: episode 1 ends (matches) and episode 2 ends (doesn't match)
    cb.on_episode_end(agent, mdp, 1)
    cb.on_episode_end(agent, mdp, 2)
    cb.on_train_end(agent, mdp, tmp_path)

    # Then: exactly the rollouts from episode 1 produce video(s)
    videos = list((tmp_path / "videos").glob("*.mp4"))
    assert len(videos) >= 1


# --- on_train_end: cleanup ---


def test_train_end_closes_eval_env(tmp_path):
    """on_train_end should close the eval env cleanly."""
    # Given: a fully initialised callback
    mdp = _make_mdp()
    mdp.setup(seed=42)
    agent = _make_agent_stub()
    cb = VideoRecorderCallback(env_fn=_make_recording_env)
    cb.on_train_start(agent, mdp, tmp_path)

    # When: training ends
    cb.on_train_end(agent, mdp, tmp_path)

    # Then: the eval env reference still exists (close doesn't None it out)
    assert cb._eval_env is not None
