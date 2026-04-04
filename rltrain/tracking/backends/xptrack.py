"""XptrackLogger — experiment tracking via the xptrack library."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class XptrackLogger:
    """MetricsLogger backend that writes to an xptrack store.

    Creates an xptrack ``Run`` on ``start()`` and logs metrics/hyperparams
    through the Run's API.  The Run is finalised on ``finish()``.

    Parameters
    ----------
    store : str | Store
        Path to the DuckDB store file, ``":memory:"`` for testing, or a
        pre-built ``xptrack.Store`` instance (e.g. ``InMemoryStore``).
    project : str
        xptrack project name.  Defaults to ``"rltrain"``.
    hooks : list[Hook] | None
        Optional xptrack lifecycle hooks forwarded to the ``Run``.
    """

    def __init__(
        self,
        store: str | Any = "experiments.duckdb",
        project: str = "rltrain",
        hooks: list[Any] | None = None,
    ) -> None:
        try:
            import xptrack  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "XptrackLogger requires the `xptrack` package. Install it with:\n  pip install xptrack"
            ) from exc
        self._store = store
        self._project = project
        self._hooks = hooks
        self._run: Any | None = None

    def start(self, config: dict[str, Any], run_dir: Path) -> None:
        """Create an xptrack Run with the experiment config."""
        import xptrack

        self._run = xptrack.Run(  # type: ignore[attr-defined]
            project=self._project,
            name=run_dir.name,
            store=self._store,
            config=config,
            tags={"run_dir": str(run_dir)},
            hooks=self._hooks,
        )
        self._run.__enter__()  # type: ignore[union-attr]

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        """Log scalar metrics to the xptrack Run."""
        if self._run is not None:
            self._run.log(metrics, step=step)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """Tag the run with hyperparameters (stringified values)."""
        if self._run is not None:
            self._run.tag({k: str(v) for k, v in params.items()})

    def finish(self) -> None:
        """Finalise the xptrack Run."""
        if self._run is not None:
            self._run.__exit__(None, None, None)
            self._run = None
