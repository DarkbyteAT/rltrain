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
        "actor": [
            {
                "fqn": "rltrain.nn.SkipMLP",
                "inputs": 4,
                "hiddens": [32, 32],
                "outputs": 2,
            }
        ],
        "critic": [
            {
                "fqn": "rltrain.nn.SkipMLP",
                "inputs": 4,
                "hiddens": [32, 32],
                "outputs": 1,
            }
        ],
    },
    "opt": {
        "actor": {"fqn": "torch.optim.Adam", "lr": 0.0003},
        "critic": {"fqn": "torch.optim.Adam", "lr": 0.001},
    },
}


def test_ppo_cartpole_smoke(tmp_path):
    agent = mk.agent(device=T.device("cpu"), **CARTPOLE_AGENT_CFG)
    env = MDP(
        mk.env(**CARTPOLE_ENV_CFG),
        run_beta=0.1,
        log_freq=100,
        swap_channels=False,
    )

    trainer = Trainer(
        agent,
        env,
        num_steps=1000,
        checkpoint_steps=500,
        run_dir=tmp_path,
        callbacks=[],
        seed=42,
    )

    trainer.fit()

    assert env.total_steps >= 1000
