"""Tests for MetricsLogger protocol structural subtyping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rltrain.tracking.logger import MetricsLogger


class _FullLogger:
    """Implements every MetricsLogger method."""

    def start(self, config: dict[str, Any], run_dir: Path) -> None: ...
    def log_scalars(self, metrics: dict[str, float], step: int) -> None: ...
    def log_hyperparams(self, params: dict[str, Any]) -> None: ...
    def finish(self) -> None: ...


class _PartialLogger:
    """Only implements log_scalars — still structurally valid because the
    protocol methods all have ``...`` defaults."""

    def start(self, config: dict[str, Any], run_dir: Path) -> None: ...
    def log_scalars(self, metrics: dict[str, float], step: int) -> None: ...
    def log_hyperparams(self, params: dict[str, Any]) -> None: ...
    def finish(self) -> None: ...


class _NotALogger:
    """Missing required method signatures."""

    def log_something(self, data: dict) -> None: ...


@pytest.mark.unit
def test_full_logger_satisfies_protocol():
    assert isinstance(_FullLogger(), MetricsLogger)


@pytest.mark.unit
def test_partial_logger_satisfies_protocol():
    assert isinstance(_PartialLogger(), MetricsLogger)


@pytest.mark.unit
def test_non_logger_rejected():
    assert not isinstance(_NotALogger(), MetricsLogger)


@pytest.mark.unit
def test_stream_logger_satisfies_protocol():
    from rltrain.tracking.backends.stream import StreamLogger

    assert isinstance(StreamLogger(), MetricsLogger)
