"""Tests that the Trainer calls callback hooks in the correct order."""

import torch as T

import rltrain.utils.builders as mk
from rltrain.env import MDP
from rltrain.trainer import Trainer


CARTPOLE_ENV_CFG = {"id": "CartPole-v1", "wrappers": []}
CARTPOLE_AGENT_CFG = {
    "fqn": "rltrain.agents.actor_critic.PPO",
    "gamma": 0.99,
    "tau": 0.01,
    "eps_per_rollout": 1,
    "normalise": False,
    "continuous": False,
    "shared_features": False,
    "beta_critic": 0.5,
    "horizon": 128,
    "lambda_gae": 0.95,
    "num_epochs": 4,
    "batch_size": 32,
    "early_stop": 0.2,
    "eps_clip": 0.2,
    "model": {
        "actor": [{"fqn": "rltrain.nn.SkipMLP", "inputs": 4, "hiddens": [32, 32], "outputs": 2}],
        "critic": [{"fqn": "rltrain.nn.SkipMLP", "inputs": 4, "hiddens": [32, 32], "outputs": 1}],
    },
    "opt": {
        "actor": {"fqn": "torch.optim.Adam", "lr": 0.0003},
        "critic": {"fqn": "torch.optim.Adam", "lr": 0.001},
    },
}


class RecordingCallback:
    """Records every hook call for assertion."""

    def __init__(self):
        self.calls: list[str] = []

    def on_train_start(self, agent, env, run_dir):
        self.calls.append("train_start")

    def on_step(self, agent, env, step):
        self.calls.append("step")

    def on_episode_end(self, agent, env, episode):
        self.calls.append("episode_end")

    def on_checkpoint(self, agent, env, run_dir):
        self.calls.append("checkpoint")

    def on_train_end(self, agent, env, run_dir):
        self.calls.append("train_end")


def test_callback_lifecycle_order(tmp_path):
    """Verify train_start is first, train_end is last, and episodes happen."""
    agent = mk.agent(device=T.device("cpu"), **CARTPOLE_AGENT_CFG)
    env = MDP(mk.env(**CARTPOLE_ENV_CFG), run_beta=0.1, log_freq=100, swap_channels=False)

    recorder = RecordingCallback()
    trainer = Trainer(
        agent, env, num_steps=500, checkpoint_steps=250,
        run_dir=tmp_path, callbacks=[recorder], seed=42,
    )
    trainer.fit()

    assert recorder.calls[0] == "train_start"
    assert recorder.calls[-1] == "train_end"
    assert "step" in recorder.calls
    assert "episode_end" in recorder.calls
    assert "checkpoint" in recorder.calls


def test_multiple_callbacks_all_called(tmp_path):
    """Verify every callback in the list receives hooks."""
    agent = mk.agent(device=T.device("cpu"), **CARTPOLE_AGENT_CFG)
    env = MDP(mk.env(**CARTPOLE_ENV_CFG), run_beta=0.1, log_freq=100, swap_channels=False)

    r1 = RecordingCallback()
    r2 = RecordingCallback()
    trainer = Trainer(
        agent, env, num_steps=500, checkpoint_steps=250,
        run_dir=tmp_path, callbacks=[r1, r2], seed=42,
    )
    trainer.fit()

    assert r1.calls == r2.calls


def test_empty_callbacks_list_runs_without_error(tmp_path):
    """Trainer with callbacks=[] should train without crashing."""
    agent = mk.agent(device=T.device("cpu"), **CARTPOLE_AGENT_CFG)
    env = MDP(mk.env(**CARTPOLE_ENV_CFG), run_beta=0.1, log_freq=100, swap_channels=False)

    trainer = Trainer(
        agent, env, num_steps=500, checkpoint_steps=250,
        run_dir=tmp_path, callbacks=[], seed=42,
    )
    trainer.fit()
    assert env.total_steps >= 500
