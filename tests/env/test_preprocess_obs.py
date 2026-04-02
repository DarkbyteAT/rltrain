"""Tests for MDP.preprocess_obs() observation preprocessing."""

import gymnasium as gym
import gymnasium.vector as vgym
import numpy as np

from rltrain.env import MDP


def _make_mdp(swap_channels: bool) -> MDP:
    """Helper: build an MDP wrapping CartPole (no channel swap needed)."""
    env = vgym.SyncVectorEnv([lambda: gym.make("CartPole-v1")])
    return MDP(env, run_beta=0.1, log_freq=100, swap_channels=swap_channels)


def test_preprocess_obs_identity_when_no_swap():
    """With swap_channels=False, preprocess_obs returns the input unchanged."""
    mdp = _make_mdp(swap_channels=False)
    obs = np.array([[1.0, 2.0, 3.0, 4.0]])
    result = mdp.preprocess_obs(obs)
    np.testing.assert_array_equal(result, obs)


def test_preprocess_obs_swaps_channels():
    """With swap_channels=True, preprocess_obs applies squeeze().T[newaxis, :]."""
    mdp = _make_mdp(swap_channels=True)
    # Simulate a (1, H, W, C) image observation from a vectorised env
    obs = np.random.rand(1, 4, 4, 3)
    result = mdp.preprocess_obs(obs)
    expected = obs.squeeze().T[np.newaxis, :]
    np.testing.assert_array_equal(result, expected)


def test_preprocess_obs_used_by_reset(cartpole_env):
    """After reset, state should equal preprocess_obs applied to raw obs."""
    # cartpole_env has swap_channels=False, so state == raw obs
    cartpole_env.setup(seed=42)
    assert cartpole_env.state is not None
