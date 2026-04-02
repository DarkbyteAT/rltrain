"""Tests for CSVLoggerCallback — happy and error paths."""

import pandas as pd

from rltrain.callbacks.csv_logger import CSVLoggerCallback


def test_on_checkpoint_writes_csv(tmp_path, stub_agent, populated_env):
    cb = CSVLoggerCallback()
    cb.on_train_start(stub_agent, populated_env, tmp_path)
    cb.on_checkpoint(stub_agent, populated_env, tmp_path)

    csv_path = tmp_path / "metrics.csv"
    assert csv_path.exists()

    df = pd.read_csv(csv_path, index_col="Episode")
    assert len(df) == populated_env.episode_count
    assert list(df.columns) == ["Length", "Return", "Running Return"]
    assert df["Return"].tolist() == populated_env.return_history


def test_on_train_end_writes_csv(tmp_path, stub_agent, populated_env):
    cb = CSVLoggerCallback()
    cb.on_train_start(stub_agent, populated_env, tmp_path)
    cb.on_train_end(stub_agent, populated_env, tmp_path)

    assert (tmp_path / "metrics.csv").exists()


def test_on_checkpoint_skips_when_zero_episodes(tmp_path, stub_agent, stub_env):
    """CSV should not be written when no episodes have completed."""
    cb = CSVLoggerCallback()
    cb.on_train_start(stub_agent, stub_env, tmp_path)
    cb.on_checkpoint(stub_agent, stub_env, tmp_path)

    assert not (tmp_path / "metrics.csv").exists()


def test_on_checkpoint_before_train_start_is_safe(stub_agent, populated_env, tmp_path):
    """Calling on_checkpoint before on_train_start should not crash."""
    cb = CSVLoggerCallback()
    cb.on_checkpoint(stub_agent, populated_env, tmp_path)


def test_csv_overwrites_on_each_checkpoint(tmp_path, stub_agent, populated_env):
    """Each checkpoint should overwrite the CSV with the full history."""
    cb = CSVLoggerCallback()
    cb.on_train_start(stub_agent, populated_env, tmp_path)

    cb.on_checkpoint(stub_agent, populated_env, tmp_path)
    df1 = pd.read_csv(tmp_path / "metrics.csv", index_col="Episode")

    populated_env.episode_count = 5
    populated_env.length_history.extend([60, 70])
    populated_env.return_history.extend([50.0, 55.0])
    populated_env.run_history.extend([43.0, 49.0])

    cb.on_checkpoint(stub_agent, populated_env, tmp_path)
    df2 = pd.read_csv(tmp_path / "metrics.csv", index_col="Episode")

    assert len(df1) == 3
    assert len(df2) == 5
