"""Tests that the Trainer calls callback hooks in the correct order."""

import pytest

from rltrain.trainer import Trainer


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


@pytest.mark.unit
def test_callback_lifecycle_order(tmp_path, cartpole_agent, cartpole_env):
    """Verify train_start is first, train_end is last, and episodes happen."""
    recorder = RecordingCallback()
    trainer = Trainer(
        cartpole_agent,
        cartpole_env,
        num_steps=500,
        checkpoint_steps=250,
        run_dir=tmp_path,
        callbacks=[recorder],
        seed=42,
    )
    trainer.fit()

    assert recorder.calls[0] == "train_start"
    assert recorder.calls[-1] == "train_end"
    assert "step" in recorder.calls
    assert "episode_end" in recorder.calls
    assert "checkpoint" in recorder.calls


@pytest.mark.unit
def test_multiple_callbacks_all_called(tmp_path, cartpole_agent, cartpole_env):
    """Verify every callback in the list receives hooks."""
    r1 = RecordingCallback()
    r2 = RecordingCallback()
    trainer = Trainer(
        cartpole_agent,
        cartpole_env,
        num_steps=500,
        checkpoint_steps=250,
        run_dir=tmp_path,
        callbacks=[r1, r2],
        seed=42,
    )
    trainer.fit()

    assert r1.calls == r2.calls


@pytest.mark.unit
def test_empty_callbacks_list_runs_without_error(tmp_path, cartpole_agent, cartpole_env):
    """Trainer with callbacks=[] should train without crashing."""
    trainer = Trainer(
        cartpole_agent,
        cartpole_env,
        num_steps=500,
        checkpoint_steps=250,
        run_dir=tmp_path,
        callbacks=[],
        seed=42,
    )
    trainer.fit()
    assert cartpole_env.total_steps >= 500
