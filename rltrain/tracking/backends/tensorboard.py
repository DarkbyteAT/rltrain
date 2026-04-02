"""TensorBoardLogger — scalar metrics and hyperparameters via SummaryWriter."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class TensorBoardLogger:
    """Wraps ``torch.utils.tensorboard.SummaryWriter``.

    Parameters
    ----------
    `log_dir`
        Directory for TensorBoard event files.  If ``None``, defaults to
        ``run_dir / "tb"`` when ``start`` is called.
    """

    def __init__(self, log_dir: str | Path | None = None) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "TensorBoardLogger requires the `tensorboard` package. Install it with:\n  pip install tensorboard"
            ) from exc
        self._log_dir = Path(log_dir) if log_dir is not None else None
        self._writer: Any | None = None

    def start(self, config: dict[str, Any], run_dir: Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        log_dir = self._log_dir or run_dir / "tb"
        self._writer = SummaryWriter(log_dir=str(log_dir))

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        if self._writer is None:
            return
        for tag, value in metrics.items():
            self._writer.add_scalar(tag, value, global_step=step)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        if self._writer is None:
            return
        self._writer.add_hparams(
            {k: v for k, v in params.items() if isinstance(v, int | float | str | bool)},
            {},
        )

    def finish(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
