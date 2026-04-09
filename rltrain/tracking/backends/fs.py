"""FSLogger — JSONL metrics to any fsspec-compatible filesystem."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FSLogger:
    """Writes one JSON object per ``log_scalars`` call to a JSONL file.

    Args:
        url: An fsspec-compatible URL (e.g. ``"s3://bucket/metrics.jsonl"``,
            ``"file:///tmp/metrics.jsonl"``, or a plain local path).
        fs_kwargs: Extra keyword arguments forwarded to ``fsspec.open``.
    """

    def __init__(self, url: str, **fs_kwargs: Any) -> None:
        """Validate that fsspec is installed and store the target URL.

        Args:
            url: fsspec-compatible URL for the output JSONL file.
            **fs_kwargs: Extra keyword arguments forwarded to ``fsspec.open``.
        """
        try:
            import fsspec as _fsspec  # noqa: F401
        except ImportError as exc:
            raise ImportError("FSLogger requires the `fsspec` package. Install it with:\n  pip install fsspec") from exc
        self._url = url
        self._fs_kwargs = fs_kwargs
        self._file: Any | None = None
        self._fh: Any | None = None

    def start(self, config: dict[str, Any], run_dir: Path) -> None:
        """Open the JSONL file via fsspec, creating it if it does not exist."""
        import fsspec

        self._file = fsspec.open(self._url, mode="w", **self._fs_kwargs)
        self._fh = self._file.open()

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        """Append a JSONL record of ``{"step": step, **metrics}`` to the file."""
        if self._fh is None:
            return
        record = {"step": step, **metrics}
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """See ``MetricsLogger.log_hyperparams``; not implemented for JSONL."""
        ...

    def finish(self) -> None:
        """Close the fsspec file handle and flush any buffered data."""
        if self._fh is not None:
            self._fh.close()
            self._file = None
            self._fh = None
