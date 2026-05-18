"""Tests for the callback protocol and built-in callbacks."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from rltrain.callbacks import Callback
from rltrain.callbacks.csv_logger import CSVLoggerCallback
from rltrain.callbacks.video_recorder import VideoRecorderCallback


@pytest.mark.unit
def test_callback_protocol_satisfaction():
    """CSVLoggerCallback and VideoRecorderCallback satisfy the Callback protocol."""
    # Given
    csv_cb = CSVLoggerCallback()
    video_cb = VideoRecorderCallback(env_fn=lambda: None)

    # Then
    assert isinstance(csv_cb, Callback)
    assert isinstance(video_cb, Callback)


@pytest.mark.unit
def test_csv_logger_writes_file(tmp_path: Path):
    """CSVLogger writes correct rows to metrics.csv after checkpoint flush."""
    # Given
    cb = CSVLoggerCallback()
    cb.on_train_start(config={}, run_dir=tmp_path)

    # When
    cb.on_episode_end(episode=1, episode_return=10.0, episode_length=50)
    cb.on_episode_end(episode=2, episode_return=20.5, episode_length=100)
    cb.on_checkpoint(step=100, agent_state=None, run_dir=tmp_path)
    cb.on_train_end(agent_state=None, run_dir=tmp_path)

    # Then
    csv_path = tmp_path / "metrics.csv"
    assert csv_path.exists()
    with open(csv_path) as f:
        reader = csv.reader(f)
        rows = list(reader)

    assert rows[0] == ["episode", "return", "length"]
    assert rows[1] == ["1", "10.0", "50"]
    assert rows[2] == ["2", "20.5", "100"]
    assert len(rows) == 3


@pytest.mark.unit
def test_csv_logger_flushes_at_checkpoint(tmp_path: Path):
    """Episodes accumulated between checkpoints are flushed at each checkpoint."""
    # Given
    cb = CSVLoggerCallback()
    cb.on_train_start(config={}, run_dir=tmp_path)

    # When -- first batch
    cb.on_episode_end(episode=1, episode_return=5.0, episode_length=25)
    cb.on_checkpoint(step=50, agent_state=None, run_dir=tmp_path)

    # When -- second batch
    cb.on_episode_end(episode=2, episode_return=15.0, episode_length=75)
    cb.on_episode_end(episode=3, episode_return=25.0, episode_length=125)
    cb.on_checkpoint(step=100, agent_state=None, run_dir=tmp_path)

    cb.on_train_end(agent_state=None, run_dir=tmp_path)

    # Then -- all three episodes present, flushed across two checkpoints
    csv_path = tmp_path / "metrics.csv"
    with open(csv_path) as f:
        reader = csv.reader(f)
        rows = list(reader)

    assert len(rows) == 4  # header + 3 episodes
    assert rows[1] == ["1", "5.0", "25"]
    assert rows[2] == ["2", "15.0", "75"]
    assert rows[3] == ["3", "25.0", "125"]


@pytest.mark.unit
def test_video_recorder_creates_directory(tmp_path: Path):
    """VideoRecorderCallback creates the video directory at train start."""
    # Given
    cb = VideoRecorderCallback()

    # When
    cb.on_train_start(config={}, run_dir=tmp_path)

    # Then
    video_dir = tmp_path / "videos"
    assert video_dir.exists()
    assert video_dir.is_dir()


@pytest.mark.unit
def test_custom_callback_all_hooks_called(tmp_path: Path):
    """A mock callback records all five hooks being called."""
    # Given
    calls: list[str] = []

    class RecordingCallback:
        def on_train_start(self, config, run_dir):
            calls.append("on_train_start")

        def on_step(self, step, metrics):
            calls.append("on_step")

        def on_episode_end(self, episode, episode_return, episode_length):
            calls.append("on_episode_end")

        def on_checkpoint(self, step, agent_state, run_dir):
            calls.append("on_checkpoint")

        def on_train_end(self, agent_state, run_dir):
            calls.append("on_train_end")

    cb = RecordingCallback()
    assert isinstance(cb, Callback)

    # When -- simulate a training loop firing all hooks
    cb.on_train_start(config={"lr": 1e-3}, run_dir=tmp_path)
    cb.on_step(step=1, metrics={"loss": 0.5})
    cb.on_episode_end(episode=1, episode_return=10.0, episode_length=50)
    cb.on_checkpoint(step=100, agent_state=None, run_dir=tmp_path)
    cb.on_train_end(agent_state=None, run_dir=tmp_path)

    # Then
    assert calls == [
        "on_train_start",
        "on_step",
        "on_episode_end",
        "on_checkpoint",
        "on_train_end",
    ]
