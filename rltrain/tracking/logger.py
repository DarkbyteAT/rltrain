"""MetricsLogger protocol — structural interface for experiment tracking backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MetricsLogger(Protocol):
    """Structural interface for experiment tracking backends.

    All methods have ``...`` defaults so that backends only need to implement the
    subset they care about.  Any class that provides the right method signatures
    is a valid ``MetricsLogger`` without explicit inheritance.
    """

    def start(self, config: dict[str, Any], run_dir: Path) -> None:
        """Called once at the beginning of training.

        Parameters
        ----------
        `config`
            Full experiment configuration dictionary.
        `run_dir`
            Filesystem path where the run's artefacts are stored.
        """
        ...

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        """Log a batch of scalar metrics at a given step.

        Parameters
        ----------
        `metrics`
            Mapping of metric name to value.
        `step`
            The global step (typically episode number).
        """
        ...

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """Record hyperparameters for the run.

        Parameters
        ----------
        `params`
            Arbitrary hyperparameter dictionary.
        """
        ...

    def finish(self) -> None:
        """Called once at the end of training to flush and close resources."""
        ...
