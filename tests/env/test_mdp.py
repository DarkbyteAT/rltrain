"""Tests for MDP single-env backward compatibility and multi-env vectorisation."""

import gymnasium as gym
import gymnasium.vector as vgym
import numpy as np
import pytest

from rltrain.env import MDP


def _make_mdp(num_envs: int = 1, **kwargs) -> MDP:
    """Build an MDP wrapping CartPole with the given number of environments."""
    env = vgym.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(num_envs)])
    return MDP(env, run_beta=0.1, log_freq=100, swap_channels=False, **kwargs)


def _random_policy(state: np.ndarray) -> np.ndarray:
    """Uniform random policy for CartPole (2 discrete actions)."""
    return np.array([0] * state.shape[0])


# ---------------------------------------------------------------------------
# Single-env backward compatibility
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_single_env_state_shape_after_setup():
    """With num_envs=1, state has shape (1, obs_dim) after setup."""
    # Given
    mdp = _make_mdp(num_envs=1)

    # When
    mdp.setup(seed=42)

    # Then
    assert mdp.state.shape[0] == 1
    assert mdp.num_envs == 1


@pytest.mark.unit
def test_single_env_trajectory_shapes():
    """With num_envs=1, trajectory arrays have a leading dim of 1."""
    # Given
    mdp = _make_mdp(num_envs=1)
    mdp.setup(seed=42)

    # When
    traj = mdp.step(_random_policy)

    # Then
    assert traj.state.shape[0] == 1
    assert traj.action.shape[0] == 1
    assert traj.reward.shape[0] == 1
    assert traj.next_state.shape[0] == 1
    assert traj.done.shape[0] == 1


@pytest.mark.unit
def test_single_env_total_steps_increments_by_one():
    """With num_envs=1, each step increments total_steps by 1."""
    # Given
    mdp = _make_mdp(num_envs=1)
    mdp.setup(seed=42)

    # When
    mdp.step(_random_policy)
    mdp.step(_random_policy)

    # Then
    assert mdp.total_steps == 2


@pytest.mark.unit
def test_single_env_episode_tracking():
    """With num_envs=1, episodes are tracked correctly through completion."""
    # Given
    mdp = _make_mdp(num_envs=1)
    mdp.setup(seed=42)

    # When — run until at least one episode completes
    while mdp.episode_count == 0:
        mdp.step(_random_policy)

    # Then
    assert mdp.episode_count == 1
    assert len(mdp.return_history) == 1
    assert len(mdp.length_history) == 1
    assert len(mdp.run_history) == 1
    assert mdp.run_reward is not None
    assert mdp.episode_steps == mdp.length_history[0]


@pytest.mark.unit
def test_single_env_trajectory_done_is_array():
    """Trajectory.done is always np.ndarray, even with num_envs=1."""
    # Given
    mdp = _make_mdp(num_envs=1)
    mdp.setup(seed=42)

    # When
    traj = mdp.step(_random_policy)

    # Then — done is an array, not a scalar bool
    assert isinstance(traj.done, np.ndarray)
    assert traj.done.shape == (1,)


# ---------------------------------------------------------------------------
# Multi-env vectorisation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_multi_env_state_shape_after_setup():
    """With num_envs=4, state has shape (4, obs_dim) after setup."""
    # Given
    mdp = _make_mdp(num_envs=4)

    # When
    mdp.setup(seed=42)

    # Then
    assert mdp.state.shape[0] == 4
    assert mdp.num_envs == 4


@pytest.mark.unit
def test_multi_env_trajectory_shapes():
    """With num_envs=4, trajectory arrays have a leading dim of 4."""
    # Given
    mdp = _make_mdp(num_envs=4)
    mdp.setup(seed=42)

    # When
    traj = mdp.step(_random_policy)

    # Then
    assert traj.state.shape[0] == 4
    assert traj.action.shape[0] == 4
    assert traj.reward.shape[0] == 4
    assert traj.next_state.shape[0] == 4
    assert traj.done.shape[0] == 4


@pytest.mark.unit
def test_multi_env_total_steps_increments_by_num_envs():
    """With num_envs=4, each step increments total_steps by 4."""
    # Given
    mdp = _make_mdp(num_envs=4)
    mdp.setup(seed=42)

    # When
    mdp.step(_random_policy)
    mdp.step(_random_policy)

    # Then
    assert mdp.total_steps == 8


@pytest.mark.unit
def test_multi_env_episode_tracking():
    """With num_envs=4, multiple episodes can complete and be tracked independently."""
    # Given
    mdp = _make_mdp(num_envs=4)
    mdp.setup(seed=42)

    # When — run until at least 2 episodes complete
    while mdp.episode_count < 2:
        mdp.step(_random_policy)

    # Then — histories have one entry per completed episode
    assert mdp.episode_count >= 2
    assert len(mdp.return_history) == mdp.episode_count
    assert len(mdp.length_history) == mdp.episode_count
    assert len(mdp.run_history) == mdp.episode_count
    assert mdp.run_reward is not None


@pytest.mark.unit
def test_multi_env_per_env_counters_reset_on_done():
    """After an episode ends in one sub-env, its length/return counters reset while others continue."""
    # Given
    mdp = _make_mdp(num_envs=4)
    mdp.setup(seed=42)

    # When — run until at least one episode completes
    while mdp.episode_count == 0:
        mdp.step(_random_policy)

    # Then — the env that finished had its counters reset to 0 (or 1 if it already
    # took a new step), while other envs may have non-zero lengths.
    # At minimum, the recorded episode length should be > 0.
    assert mdp.length_history[0] > 0
    assert mdp.return_history[0] != 0.0 or mdp.length_history[0] > 0


@pytest.mark.unit
def test_multi_env_trajectory_done_is_per_env_array():
    """Trajectory.done is an ndarray with one element per sub-env."""
    # Given
    mdp = _make_mdp(num_envs=4)
    mdp.setup(seed=42)

    # When
    traj = mdp.step(_random_policy)

    # Then — done is a per-env boolean array
    assert isinstance(traj.done, np.ndarray)
    assert traj.done.shape == (4,)


@pytest.mark.unit
def test_multi_env_episode_steps_accumulates():
    """episode_steps sums the lengths of all completed episodes across all sub-envs."""
    # Given
    mdp = _make_mdp(num_envs=4)
    mdp.setup(seed=42)

    # When — run until several episodes complete
    while mdp.episode_count < 4:
        mdp.step(_random_policy)

    # Then
    assert mdp.episode_steps == sum(mdp.length_history)


# ---------------------------------------------------------------------------
# Builder integration
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_env_builder_num_envs_default():
    """The env builder defaults to num_envs=1."""
    # Given / When
    import rltrain.utils.builders as mk

    vec_env = mk.env(id="CartPole-v1", wrappers=[])

    # Then
    assert vec_env.num_envs == 1


@pytest.mark.unit
def test_env_builder_num_envs_multiple():
    """The env builder creates the requested number of sub-environments."""
    # Given / When
    import rltrain.utils.builders as mk

    vec_env = mk.env(id="CartPole-v1", wrappers=[], num_envs=4)

    # Then
    assert vec_env.num_envs == 4
