"""Smoke tests for the Trainer."""

import torch as T

import rltrain.utils.builders as mk
from rltrain.callbacks.checkpoint import CheckpointCallback
from rltrain.callbacks.csv_logger import CSVLoggerCallback
from rltrain.callbacks.plot import PlotCallback
from rltrain.env import MDP
from rltrain.trainer import Trainer


def test_ppo_cartpole_smoke(tmp_path, cartpole_agent, cartpole_env):
    """PPO trains on CartPole for 1000 steps without crashing."""
    trainer = Trainer(
        cartpole_agent,
        cartpole_env,
        num_steps=1000,
        checkpoint_steps=500,
        run_dir=tmp_path,
        callbacks=[],
        seed=42,
    )
    trainer.fit()

    assert cartpole_env.total_steps >= 1000
    assert cartpole_env.episode_count > 0
    assert len(cartpole_env.return_history) == cartpole_env.episode_count


def test_trainer_default_callbacks(tmp_path, cartpole_agent, cartpole_env):
    """Trainer with callbacks=None should use the 3 built-in defaults."""
    trainer = Trainer(
        cartpole_agent,
        cartpole_env,
        num_steps=500,
        checkpoint_steps=250,
        run_dir=tmp_path,
        seed=42,
    )

    assert len(trainer.callbacks) == 3
    assert any(isinstance(cb, CSVLoggerCallback) for cb in trainer.callbacks)
    assert any(isinstance(cb, PlotCallback) for cb in trainer.callbacks)
    assert any(isinstance(cb, CheckpointCallback) for cb in trainer.callbacks)


def test_ppo_multi_env_cartpole_e2e(tmp_path):
    """PPO trains on CartPole with 4 parallel envs end-to-end."""
    # Given — 4-env vectorised setup
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
                "actor": [{"fqn": "toblox.SkipMLP", "inputs": 4, "hiddens": [32], "outputs": 2}],
                "critic": [{"fqn": "toblox.SkipMLP", "inputs": 4, "hiddens": [32], "outputs": 1}],
            },
            "opt": {
                "actor": {"fqn": "torch.optim.Adam", "lr": 3e-4},
                "critic": {"fqn": "torch.optim.Adam", "lr": 1e-3},
            },
        },
    )

    trainer = Trainer(
        agent,
        mdp,
        num_steps=1000,
        checkpoint_steps=500,
        run_dir=tmp_path,
        callbacks=[
            CSVLoggerCallback(),
            CheckpointCallback(),
        ],
        seed=42,
    )

    # When
    trainer.fit()

    # Then — trained successfully with multi-env
    assert mdp.num_envs == 4
    assert mdp.total_steps >= 1000
    assert mdp.episode_count > 0
    assert len(mdp.return_history) == mdp.episode_count
    # CSV and checkpoint files created
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "models" / "model_FINAL.pt").exists()
