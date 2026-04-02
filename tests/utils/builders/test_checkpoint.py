"""Tests for round-trip save/load of agent checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch as T

import rltrain.utils.builders as mk
from rltrain.utils.builders.checkpoint import load_agent
from tests.conftest import CARTPOLE_AGENT_CFG, DQN_AGENT_CFG


def _build_and_setup_agent():
    """Build a minimal PPO agent on CPU and call setup()."""
    agent = mk.agent(device=T.device("cpu"), **CARTPOLE_AGENT_CFG)
    agent.setup()
    return agent


def _save_run_dir(tmp_path: Path, agent, *, checkpoint: str = "FINAL") -> Path:
    """Persist an agent's config and weights into a fake run directory."""
    run_dir = tmp_path / "run"
    (run_dir / "config").mkdir(parents=True)
    (run_dir / "models").mkdir(parents=True)

    with open(run_dir / "config" / "agent.json", "w") as f:
        json.dump(CARTPOLE_AGENT_CFG, f)

    T.save(agent.model.state_dict(), run_dir / "models" / f"model_{checkpoint}.pt")
    return run_dir


def test_round_trip_weights_match(tmp_path: Path):
    """Loaded agent's parameters must be identical to the original's."""
    # Given -- a trained agent saved to disk
    original = _build_and_setup_agent()
    run_dir = _save_run_dir(tmp_path, original)

    # When -- we load it back
    loaded = load_agent(run_dir, device="cpu")

    # Then -- every parameter tensor matches exactly
    for (name, p_orig), (_, p_load) in zip(
        original.model.named_parameters(), loaded.model.named_parameters(), strict=True
    ):
        assert T.equal(p_orig, p_load), f"parameter {name!r} differs after round-trip"


def test_round_trip_same_output(tmp_path: Path):
    """Loaded agent must produce the same action distribution as the original."""
    # Given -- a saved agent and a batch of states
    original = _build_and_setup_agent()
    original.model.eval()
    run_dir = _save_run_dir(tmp_path, original)

    states = T.randn(4, 4)  # batch of 4 CartPole observations

    # When -- we load and run both agents
    with T.inference_mode():
        original_out = original.act(states.to(original.device))
        loaded = load_agent(run_dir, device="cpu")
        loaded_out = loaded.act(states.to(loaded.device))

    # Then -- logits (Categorical) or loc (Normal) must match
    assert T.allclose(original_out.logits, loaded_out.logits, atol=1e-6), "action distribution differs after round-trip"


def test_loaded_model_in_eval_mode(tmp_path: Path):
    """load_agent must set the model to eval mode."""
    # Given -- a saved agent
    original = _build_and_setup_agent()
    run_dir = _save_run_dir(tmp_path, original)

    # When
    loaded = load_agent(run_dir, device="cpu")

    # Then
    assert not loaded.model.training, "model should be in eval mode after loading"


def test_step_based_checkpoint(tmp_path: Path):
    """load_agent must support step-based checkpoint names like '2500'."""
    # Given -- a saved agent with a step-based checkpoint
    original = _build_and_setup_agent()
    run_dir = _save_run_dir(tmp_path, original, checkpoint="2500")

    # When
    loaded = load_agent(run_dir, checkpoint="2500", device="cpu")

    # Then -- weights match
    for (name, p_orig), (_, p_load) in zip(
        original.model.named_parameters(), loaded.model.named_parameters(), strict=True
    ):
        assert T.equal(p_orig, p_load), f"parameter {name!r} differs for step checkpoint"


def test_missing_config_raises(tmp_path: Path):
    """load_agent must raise FileNotFoundError when agent.json is missing."""
    # Given -- a run directory with no config
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()

    # When / Then
    with pytest.raises(FileNotFoundError, match="Agent config not found"):
        load_agent(run_dir, device="cpu")


def test_missing_checkpoint_raises(tmp_path: Path):
    """load_agent must raise FileNotFoundError when the .pt file is missing."""
    # Given -- a run directory with config but no model file
    _build_and_setup_agent()  # just to verify the config is valid
    run_dir = tmp_path / "run"
    (run_dir / "config").mkdir(parents=True)
    (run_dir / "models").mkdir(parents=True)

    with open(run_dir / "config" / "agent.json", "w") as f:
        json.dump(CARTPOLE_AGENT_CFG, f)

    # When / Then
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_agent(run_dir, device="cpu")


def test_callable_produces_actions(tmp_path: Path):
    """The loaded agent's __call__ must return numpy actions."""
    # Given -- a saved and reloaded agent
    original = _build_and_setup_agent()
    run_dir = _save_run_dir(tmp_path, original)

    loaded = load_agent(run_dir, device="cpu")

    # When -- called with a numpy observation (batch of 1)
    obs = np.random.randn(1, 4).astype(np.float32)
    actions = loaded(obs)

    # Then -- returns a numpy array with correct shape
    assert isinstance(actions, np.ndarray)
    assert actions.shape == (1,), f"expected shape (1,), got {actions.shape}"


def test_accepts_string_path(tmp_path: Path):
    """load_agent must accept a string path, not just a Path object."""
    # Given
    original = _build_and_setup_agent()
    run_dir = _save_run_dir(tmp_path, original)

    # When / Then -- no TypeError
    loaded = load_agent(str(run_dir), device="cpu")
    assert not loaded.model.training


def test_dqn_round_trip(tmp_path: Path):
    """DQN agent round-trip: weights, target sync, eps_greedy, and act()."""
    # Given -- a DQN agent saved to disk
    original = mk.agent(device=T.device("cpu"), **DQN_AGENT_CFG)
    original.setup()

    run_dir = tmp_path / "run"
    (run_dir / "config").mkdir(parents=True)
    (run_dir / "models").mkdir(parents=True)

    with open(run_dir / "config" / "agent.json", "w") as f:
        json.dump(DQN_AGENT_CFG, f)

    T.save(original.model.state_dict(), run_dir / "models" / "model_FINAL.pt")

    # When -- we load it back
    loaded = load_agent(run_dir, device="cpu")

    # Then -- qnet weights match the original
    for (name, p_orig), (_, p_load) in zip(
        original.model["qnet"].named_parameters(),
        loaded.model["qnet"].named_parameters(),
        strict=True,
    ):
        assert T.equal(p_orig, p_load), f"qnet parameter {name!r} differs after round-trip"

    # Then -- target network is synced to the loaded qnet (not stale from setup)
    for (name, p_qnet), (_, p_target) in zip(
        loaded.model["qnet"].named_parameters(),
        loaded.target.named_parameters(),
        strict=True,
    ):
        assert T.equal(p_qnet, p_target), f"target parameter {name!r} not synced to qnet"

    # Then -- eps_greedy is 0.0 (inference mode, not random actions)
    assert loaded.eps_greedy == 0.0, f"expected eps_greedy=0.0, got {loaded.eps_greedy}"

    # Then -- act() produces valid actions
    obs = np.random.randn(1, 4).astype(np.float32)
    actions = loaded(obs)
    assert isinstance(actions, np.ndarray)
    assert actions.shape == (1,), f"expected shape (1,), got {actions.shape}"


def test_invalid_checkpoint_name_raises():
    """load_agent must reject checkpoint names that are not FINAL or numeric."""
    # When / Then
    with pytest.raises(ValueError, match="checkpoint must be 'FINAL' or a numeric step"):
        load_agent("/nonexistent", checkpoint="../../../etc/passwd", device="cpu")
