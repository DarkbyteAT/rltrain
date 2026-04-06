"""Tests for CSVLoggerCallback — happy and error paths."""

import pandas as pd
import pytest

from rltrain.callbacks.csv_logger import CSVLoggerCallback


@pytest.mark.unit
def test_on_checkpoint_writes_csv(tmp_path, stub_agent, populated_env):
    cb = CSVLoggerCallback()
    cb.on_train_start(stub_agent, populated_env, tmp_path)
    cb.on_checkpoint(stub_agent, populated_env, tmp_path)

    csv_path = tmp_path / "metrics.csv"
    assert csv_path.exists()

    df = pd.read_csv(csv_path, index_col="episode")
    assert len(df) == populated_env.episode_count
    assert list(df.columns) == ["length", "return", "running_return"]
    assert df["return"].tolist() == populated_env.return_history


@pytest.mark.unit
def test_on_train_end_writes_csv(tmp_path, stub_agent, populated_env):
    cb = CSVLoggerCallback()
    cb.on_train_start(stub_agent, populated_env, tmp_path)
    cb.on_train_end(stub_agent, populated_env, tmp_path)

    df = pd.read_csv(tmp_path / "metrics.csv", index_col="episode")
    assert len(df) == populated_env.episode_count


@pytest.mark.unit
def test_on_checkpoint_skips_when_zero_episodes(tmp_path, stub_agent, stub_env):
    """CSV should not be written when no episodes have completed."""
    cb = CSVLoggerCallback()
    cb.on_train_start(stub_agent, stub_env, tmp_path)
    cb.on_checkpoint(stub_agent, stub_env, tmp_path)

    assert not (tmp_path / "metrics.csv").exists()


@pytest.mark.unit
def test_on_checkpoint_before_train_start_is_safe(stub_agent, populated_env, tmp_path):
    """Calling on_checkpoint before on_train_start should not crash or write files."""
    cb = CSVLoggerCallback()
    cb.on_checkpoint(stub_agent, populated_env, tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
def test_csv_overwrites_on_each_checkpoint(tmp_path, stub_agent, populated_env):
    """Each checkpoint should overwrite the CSV with the full history."""
    cb = CSVLoggerCallback()
    cb.on_train_start(stub_agent, populated_env, tmp_path)

    cb.on_checkpoint(stub_agent, populated_env, tmp_path)
    df1 = pd.read_csv(tmp_path / "metrics.csv", index_col="episode")

    populated_env.episode_count = 5
    populated_env.length_history.extend([60, 70])
    populated_env.return_history.extend([50.0, 55.0])
    populated_env.run_history.extend([43.0, 49.0])

    cb.on_checkpoint(stub_agent, populated_env, tmp_path)
    df2 = pd.read_csv(tmp_path / "metrics.csv", index_col="episode")

    assert len(df1) == 3
    assert len(df2) == 5
