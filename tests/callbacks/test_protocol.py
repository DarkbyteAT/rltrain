"""Tests for the Callback protocol — verifies structural subtyping works."""

from rltrain.callbacks import Callback
from rltrain.callbacks.checkpoint import CheckpointCallback
from rltrain.callbacks.csv_logger import CSVLoggerCallback
from rltrain.callbacks.plot import PlotCallback
from rltrain.callbacks.video_recorder import VideoRecorderCallback


def test_checkpoint_satisfies_protocol():
    assert isinstance(CheckpointCallback(), Callback)


def test_csv_logger_satisfies_protocol():
    assert isinstance(CSVLoggerCallback(), Callback)


def test_plot_satisfies_protocol():
    assert isinstance(PlotCallback(num_steps=1000), Callback)


def test_custom_callback_satisfies_protocol():
    """A user-defined class with the right methods should satisfy Callback."""

    class MyCallback:
        def on_train_start(self, agent, env, run_dir): ...
        def on_step(self, agent, env, step): ...
        def on_episode_end(self, agent, env, episode): ...
        def on_checkpoint(self, agent, env, run_dir): ...
        def on_train_end(self, agent, env, run_dir): ...

    assert isinstance(MyCallback(), Callback)


def test_partial_callback_does_not_satisfy_protocol():
    """A class missing hook methods should NOT satisfy the protocol."""

    class IncompleteCallback:
        def on_train_start(self, agent, env, run_dir): ...

    assert not isinstance(IncompleteCallback(), Callback)


def test_video_recorder_satisfies_protocol():
    assert isinstance(VideoRecorderCallback(), Callback)
