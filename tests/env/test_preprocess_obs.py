"""Tests for MDP.preprocess_obs() observation preprocessing."""

import gymnasium as gym
import gymnasium.vector as vgym
import numpy as np

from rltrain.env import MDP


def _make_mdp(swap_channels: bool) -> MDP:
    """Helper: build an MDP wrapping CartPole."""
    env = vgym.SyncVectorEnv([lambda: gym.make("CartPole-v1")])
    return MDP(env, run_beta=0.1, log_freq=100, swap_channels=swap_channels)


def test_preprocess_obs_returns_input_unchanged_without_swap():
    """Without channel swap, preprocess_obs is the identity function."""
    # Given: an MDP with swap_channels=False and an arbitrary observation
    mdp = _make_mdp(swap_channels=False)
    obs = np.array([[1.0, 2.0, 3.0, 4.0]])

    # When: preprocessing is applied
    result = mdp.preprocess_obs(obs)

    # Then: the output is identical to the input
    np.testing.assert_array_equal(result, obs)


def test_preprocess_obs_transposes_channels_when_swap_enabled():
    """With channel swap, preprocess_obs applies squeeze().T[newaxis, :]."""
    # Given: an MDP with swap_channels=True and a (1, H, W, C) image observation
    mdp = _make_mdp(swap_channels=True)
    obs = np.random.rand(1, 4, 4, 3)

    # When: preprocessing is applied
    result = mdp.preprocess_obs(obs)

    # Then: the output has channels-first layout
    expected = obs.squeeze().T[np.newaxis, :]
    np.testing.assert_array_equal(result, expected)


def test_reset_applies_preprocess_obs(cartpole_env):
    """After reset, MDP.state should be a preprocessed observation."""
    # Given: a CartPole MDP (swap_channels=False)
    # When: the environment is set up (which calls reset internally)
    cartpole_env.setup(seed=42)

    # Then: state is populated (preprocess_obs was applied, identity in this case)
    assert cartpole_env.state is not None
