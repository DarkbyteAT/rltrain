"""User story tests for multi-env vectorisation.

These tests verify the high-level behaviours that multi-env enables,
rather than testing internal implementation details.
"""

import numpy as np
import torch as T

import rltrain.utils.builders as mk
from rltrain.env import MDP
from rltrain.trainer import Trainer


def _random_policy(state: np.ndarray) -> np.ndarray:
    return np.array([0] * state.shape[0])


# ---------------------------------------------------------------------------
# Story: multi-env collects data faster (more steps per wall-clock step)
# ---------------------------------------------------------------------------


def test_multi_env_collects_num_envs_steps_per_call():
    """With num_envs=4, one MDP.step() call advances 4 environment steps."""
    # Given
    vec_env = mk.env(id="CartPole-v1", wrappers=[], num_envs=4)
    mdp = MDP(vec_env, run_beta=0.1, log_freq=100, swap_channels=False)
    mdp.setup(seed=42)

    # When
    mdp.step(_random_policy)

    # Then — 4 envs each took 1 step = 4 total steps
    assert mdp.total_steps == 4


# ---------------------------------------------------------------------------
# Story: episode tracking is correct across parallel envs
# ---------------------------------------------------------------------------


def test_multi_env_tracks_all_completed_episodes():
    """Every episode that completes in any sub-env is recorded in the history."""
    # Given
    vec_env = mk.env(id="CartPole-v1", wrappers=[], num_envs=4)
    mdp = MDP(vec_env, run_beta=0.1, log_freq=100, swap_channels=False)
    mdp.setup(seed=42)

    # When — run until at least 8 episodes complete
    while mdp.episode_count < 8:
        mdp.step(_random_policy)

    # Then — every completed episode has a history entry
    assert len(mdp.return_history) == mdp.episode_count
    assert len(mdp.length_history) == mdp.episode_count
    assert len(mdp.run_history) == mdp.episode_count
    assert mdp.episode_steps == sum(mdp.length_history)


# ---------------------------------------------------------------------------
# Story: Trainer fires on_episode_end for every completed episode
# ---------------------------------------------------------------------------


class EpisodeCounter:
    """Callback that counts on_episode_end calls and records episode numbers."""

    def __init__(self):
        self.episodes: list[int] = []

    def on_train_start(self, agent, env, run_dir): ...
    def on_step(self, agent, env, step): ...
    def on_checkpoint(self, agent, env, run_dir): ...
    def on_train_end(self, agent, env, run_dir): ...

    def on_episode_end(self, agent, env, episode):
        self.episodes.append(episode)


def test_trainer_fires_episode_end_per_episode_with_multi_env(tmp_path):
    """With num_envs=4, each completed episode gets its own on_episode_end call."""
    # Given
    vec_env = mk.env(id="CartPole-v1", wrappers=[], num_envs=4)
    mdp = MDP(vec_env, run_beta=0.1, log_freq=100, swap_channels=False)
    agent = mk.agent(
        device=T.device("cpu"),
        **{
            "fqn": "rltrain.agents.actor_critic.PPO",
            "gamma": 0.99,
            "tau": 0.01,
            "normalise": False,
            "continuous": False,
            "shared_features": False,
            "beta_critic": 0.5,
            "horizon": 64,
            "lambda_gae": 0.95,
            "num_epochs": 2,
            "batch_size": 32,
            "early_stop": 0.2,
            "eps_clip": 0.2,
            "model": {
                "actor": [{"fqn": "rltrain.nn.SkipMLP", "inputs": 4, "hiddens": [32], "outputs": 2}],
                "critic": [{"fqn": "rltrain.nn.SkipMLP", "inputs": 4, "hiddens": [32], "outputs": 1}],
            },
            "opt": {
                "actor": {"fqn": "torch.optim.Adam", "lr": 3e-4},
                "critic": {"fqn": "torch.optim.Adam", "lr": 1e-3},
            },
        },
    )
    counter = EpisodeCounter()

    trainer = Trainer(
        agent,
        mdp,
        num_steps=500,
        checkpoint_steps=250,
        run_dir=tmp_path,
        callbacks=[counter],
        seed=42,
    )

    # When
    trainer.fit()

    # Then — episode numbers are consecutive (1, 2, 3, ...) with no gaps
    assert len(counter.episodes) == mdp.episode_count
    assert counter.episodes == list(range(1, mdp.episode_count + 1))


# ---------------------------------------------------------------------------
# Story: single-env backward compatibility
# ---------------------------------------------------------------------------


def test_single_env_trainer_still_works(tmp_path):
    """num_envs=1 (default) produces the same behaviour as before multi-env."""
    # Given — standard single-env setup
    vec_env = mk.env(id="CartPole-v1", wrappers=[])
    mdp = MDP(vec_env, run_beta=0.1, log_freq=100, swap_channels=False)
    agent = mk.agent(
        device=T.device("cpu"),
        **{
            "fqn": "rltrain.agents.actor_critic.PPO",
            "gamma": 0.99,
            "tau": 0.01,
            "normalise": False,
            "continuous": False,
            "shared_features": False,
            "beta_critic": 0.5,
            "horizon": 64,
            "lambda_gae": 0.95,
            "num_epochs": 2,
            "batch_size": 32,
            "early_stop": 0.2,
            "eps_clip": 0.2,
            "model": {
                "actor": [{"fqn": "rltrain.nn.SkipMLP", "inputs": 4, "hiddens": [32], "outputs": 2}],
                "critic": [{"fqn": "rltrain.nn.SkipMLP", "inputs": 4, "hiddens": [32], "outputs": 1}],
            },
            "opt": {
                "actor": {"fqn": "torch.optim.Adam", "lr": 3e-4},
                "critic": {"fqn": "torch.optim.Adam", "lr": 1e-3},
            },
        },
    )
    counter = EpisodeCounter()

    trainer = Trainer(
        agent,
        mdp,
        num_steps=500,
        checkpoint_steps=250,
        run_dir=tmp_path,
        callbacks=[counter],
        seed=42,
    )

    # When
    trainer.fit()

    # Then — episodes are consecutive, total_steps matches expectation
    assert mdp.total_steps >= 500
    assert mdp.num_envs == 1
    assert counter.episodes == list(range(1, mdp.episode_count + 1))


# ---------------------------------------------------------------------------
# Story: env builder supports num_envs parameter
# ---------------------------------------------------------------------------


def test_env_builder_creates_requested_envs():
    """mk.env(num_envs=N) creates a SyncVectorEnv with N sub-environments."""
    # Given / When
    vec_env = mk.env(id="CartPole-v1", wrappers=[], num_envs=8)

    # Then
    assert vec_env.num_envs == 8


def test_env_builder_default_is_single_env():
    """mk.env() without num_envs defaults to 1."""
    # Given / When
    vec_env = mk.env(id="CartPole-v1", wrappers=[])

    # Then
    assert vec_env.num_envs == 1
