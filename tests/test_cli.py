"""Smoke tests for the rltrain CLI and builder system."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import pytest


@pytest.mark.unit
def test_builder_constructs_ppo_from_config():
    """The agent builder resolves a PPO JSON config into a runnable module."""
    import rltrain.builders as mk

    cfg = {
        "fqn": "rltrain.agents.PPO",
        "gamma": 0.99,
        "tau": 0.0,
        "beta_critic": 0.5,
        "lambda_gae": 0.95,
        "eps_clip": 0.2,
        "num_epochs": 2,
        "minibatch_size": 8,
        "actor": {"fqn": "rltrain.networks.MLP", "in_size": 4, "out_size": 16, "width_size": 16, "depth": 1},
        "critic": {"fqn": "rltrain.networks.MLP", "in_size": 4, "out_size": 1, "width_size": 16, "depth": 1},
        "action_head": {"fqn": "rltrain.heads.DiscreteHead", "feature_dim": 16, "action_dim": 2},
        "optimizer": {"fqn": "optax.adam", "learning_rate": 3e-4},
    }
    agent = mk.agent(key=jax.random.PRNGKey(0), **cfg)

    assert type(agent).__name__ == "PPO"
    assert agent.gamma == 0.99
    assert type(agent.actor).__name__ == "MLP"


@pytest.mark.unit
def test_builder_env_dispatches_on_backend():
    """The env builder selects GymnaxEnv or GymnasiumEnv from the backend kwarg."""
    import rltrain.builders as mk

    gymnax_env = mk.env(id="CartPole-v1", backend="gymnax")
    gymnasium_env = mk.env(id="CartPole-v1", backend="gymnasium")

    assert type(gymnax_env).__name__ == "GymnaxEnv"
    assert type(gymnasium_env).__name__ == "GymnasiumEnv"


@pytest.mark.integration
def test_cli_smoke(tmp_path: Path):
    """The CLI runs a tiny PPO/CartPole job and produces the expected artifacts."""
    from typer.testing import CliRunner

    from rltrain.cli import app

    agent_path = tmp_path / "agent.json"
    env_path = tmp_path / "env.json"
    dump_path = tmp_path / "out"

    agent_path.write_text(
        json.dumps(
            {
                "fqn": "rltrain.agents.PPO",
                "gamma": 0.99,
                "tau": 0.0,
                "beta_critic": 0.5,
                "lambda_gae": 0.95,
                "eps_clip": 0.2,
                "num_epochs": 2,
                "minibatch_size": 8,
                "actor": {"fqn": "rltrain.networks.MLP", "in_size": 4, "out_size": 16, "width_size": 16, "depth": 1},
                "critic": {"fqn": "rltrain.networks.MLP", "in_size": 4, "out_size": 1, "width_size": 16, "depth": 1},
                "action_head": {"fqn": "rltrain.heads.DiscreteHead", "feature_dim": 16, "action_dim": 2},
                "optimizer": {"fqn": "optax.adam", "learning_rate": 3e-4},
            }
        )
    )
    env_path.write_text(json.dumps({"id": "CartPole-v1", "backend": "gymnax", "reward_run_rate": 0.1}))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--agent",
            str(agent_path),
            "--env",
            str(env_path),
            "--dump",
            str(dump_path),
            "--num-steps",
            "256",
            "--checkpoint-steps",
            "128",
            "--seed",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    runs = list(dump_path.glob("PPO/*/models/model_FINAL.eqx"))
    assert runs, "no FINAL checkpoint produced"
