"""XptrackLogger — experiment tracking via the xptrack client (stub)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class XptrackLogger:
    """Thin wrapper around the ``xptrack`` client.

    This is a stub implementation — ``xptrack`` is still under active
    development in a sibling repository.

    Parameters
    ----------
    `connection_kwargs`
        Keyword arguments forwarded to the xptrack client constructor.
    """

    def __init__(self, **connection_kwargs: Any) -> None:
        try:
            import xptrack as _xptrack  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "XptrackLogger requires the `xptrack` package. Install it with:\n  pip install xptrack"
            ) from exc
        self._connection_kwargs = connection_kwargs
        self._client: Any | None = None

    def start(self, config: dict[str, Any], run_dir: Path) -> None: ...

    def log_scalars(self, metrics: dict[str, float], step: int) -> None: ...

    def log_hyperparams(self, params: dict[str, Any]) -> None: ...

    def finish(self) -> None: ...
