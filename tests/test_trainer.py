"""Smoke tests for the Trainer."""

from rltrain.callbacks.checkpoint import CheckpointCallback
from rltrain.callbacks.csv_logger import CSVLoggerCallback
from rltrain.callbacks.plot import PlotCallback
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
