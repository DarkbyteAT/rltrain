"""Tests for StreamLogger output formatting."""

from __future__ import annotations

import io
from pathlib import Path

from rltrain.tracking.backends.stream import StreamLogger


def test_log_scalars_writes_formatted_line():
    buf = io.StringIO()
    logger = StreamLogger(stream=buf)
    logger.start({}, Path("/tmp"))

    logger.log_scalars({"return": 15.3, "length": 120.0, "running_return": 12.8}, step=42)

    output = buf.getvalue()
    assert output.startswith("[step 42]")
    assert "return=15.3" in output
    assert "length=120" in output
    assert "running_return=12.8" in output
    assert output.endswith("\n")


def test_log_scalars_multiple_steps():
    buf = io.StringIO()
    logger = StreamLogger(stream=buf)

    logger.log_scalars({"x": 1.0}, step=1)
    logger.log_scalars({"x": 2.0}, step=2)

    lines = buf.getvalue().strip().split("\n")
    assert len(lines) == 2
    assert "[step 1]" in lines[0]
    assert "[step 2]" in lines[1]


def test_defaults_to_stdout(capsys):
    logger = StreamLogger()

    logger.log_scalars({"val": 3.14}, step=0)

    captured = capsys.readouterr()
    assert "[step 0] val=3.14" in captured.out


def test_start_and_finish_are_noops():
    buf = io.StringIO()
    logger = StreamLogger(stream=buf)

    logger.start({"key": "value"}, Path("/tmp"))
    logger.finish()

    assert buf.getvalue() == ""


def test_log_hyperparams_is_noop():
    buf = io.StringIO()
    logger = StreamLogger(stream=buf)

    logger.log_hyperparams({"lr": 0.001})

    assert buf.getvalue() == ""
