"""Tests for CheckpointCallback — happy and error paths."""

import torch as T

from rltrain.callbacks.checkpoint import CheckpointCallback


def test_on_train_end_saves_final_model(tmp_path, stub_agent, stub_env):
    cb = CheckpointCallback()
    cb.on_train_start(stub_agent, stub_env, tmp_path)
    cb.on_train_end(stub_agent, stub_env, tmp_path)

    final_path = tmp_path / "models" / "model_FINAL.pt"
    assert final_path.exists()

    state_dict = T.load(final_path, weights_only=True)
    assert "actor.0.weight" in state_dict


def test_on_checkpoint_saves_intermediate_when_save_all(tmp_path, stub_agent, populated_env):
    cb = CheckpointCallback(save_all=True)
    cb.on_train_start(stub_agent, populated_env, tmp_path)
    cb.on_checkpoint(stub_agent, populated_env, tmp_path)

    checkpoint_path = tmp_path / "models" / f"model_{populated_env.episode_steps}.pt"
    assert checkpoint_path.exists()


def test_on_checkpoint_skips_when_not_save_all(tmp_path, stub_agent, populated_env):
    cb = CheckpointCallback(save_all=False)
    cb.on_train_start(stub_agent, populated_env, tmp_path)
    cb.on_checkpoint(stub_agent, populated_env, tmp_path)

    models_dir = tmp_path / "models"
    assert models_dir.exists()
    assert list(models_dir.iterdir()) == []


def test_on_train_start_creates_models_dir(tmp_path, stub_agent, stub_env):
    cb = CheckpointCallback()
    models_dir = tmp_path / "models"
    assert not models_dir.exists()

    cb.on_train_start(stub_agent, stub_env, tmp_path)
    assert models_dir.exists()


def test_on_checkpoint_before_train_start_is_safe(stub_agent, populated_env, tmp_path):
    """Calling on_checkpoint before on_train_start should not crash or write files."""
    cb = CheckpointCallback(save_all=True)
    cb.on_checkpoint(stub_agent, populated_env, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_on_train_end_before_train_start_is_safe(stub_agent, stub_env, tmp_path):
    """Calling on_train_end before on_train_start should not crash or write files."""
    cb = CheckpointCallback()
    cb.on_train_end(stub_agent, stub_env, tmp_path)
    assert list(tmp_path.iterdir()) == []
