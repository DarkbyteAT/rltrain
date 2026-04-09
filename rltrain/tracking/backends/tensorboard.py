"""TensorBoardLogger — scalar metrics and hyperparameters via SummaryWriter."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class TensorBoardLogger:
    """Wraps ``torch.utils.tensorboard.SummaryWriter``.

    Args:
        log_dir: Directory for TensorBoard event files.  If ``None``, defaults to
            ``run_dir / "tb"`` when ``start`` is called.
    """

    def __init__(self, log_dir: str | Path | None = None) -> None:
        """Validate that tensorboard is installed and store the log directory.

        Args:
            log_dir: Directory for TensorBoard event files; defaults to ``run_dir / "tb"`` at start time.
        """
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "TensorBoardLogger requires the `tensorboard` package. Install it with:\n  pip install tensorboard"
            ) from exc
        self._log_dir = Path(log_dir) if log_dir is not None else None
        self._writer: Any | None = None

    def start(self, config: dict[str, Any], run_dir: Path) -> None:
        """Open a ``SummaryWriter`` in ``log_dir`` (or ``run_dir / "tb"`` if unset)."""
        from torch.utils.tensorboard import SummaryWriter

        log_dir = self._log_dir or run_dir / "tb"
        self._writer = SummaryWriter(log_dir=str(log_dir))

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        """Write each metric as a TensorBoard scalar at ``global_step=step``."""
        if self._writer is None:
            return
        for tag, value in metrics.items():
            self._writer.add_scalar(tag, value, global_step=step)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """Record scalar/string hyperparameters via the TensorBoard HParams plugin."""
        if self._writer is None:
            return
        # TensorBoard's HParams plugin requires at least one metric for the run
        # to appear in the UI. Use a placeholder that will be overwritten by
        # real metrics as training progresses.
        hparam_dict = {k: v for k, v in params.items() if isinstance(v, int | float | str | bool)}
        self._writer.add_hparams(hparam_dict, {"hp/placeholder": 0.0})

    def finish(self) -> None:
        """Close the ``SummaryWriter`` and flush pending events."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
