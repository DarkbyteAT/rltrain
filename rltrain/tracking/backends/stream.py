"""StreamLogger — human-readable metrics to any text stream (zero dependencies)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import IO, Any


class StreamLogger:
    """Writes human-readable metric lines to a text stream.

    Parameters
    ----------
    `stream`
        Writable text stream.  Defaults to ``sys.stdout``.
    """

    def __init__(self, stream: IO[str] | None = None) -> None:
        self._stream = stream or sys.stdout

    def start(self, config: dict[str, Any], run_dir: Path) -> None: ...

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        parts = " ".join(f"{k}={v:.4g}" for k, v in metrics.items())
        self._stream.write(f"[step {step}] {parts}\n")
        self._stream.flush()

    def log_hyperparams(self, params: dict[str, Any]) -> None: ...

    def finish(self) -> None: ...
