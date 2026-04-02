"""WandbLogger — experiment tracking via Weights & Biases."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class WandbLogger:
    """Wraps ``wandb.init``, ``wandb.log``, ``wandb.config.update``, and ``wandb.finish``.

    Parameters
    ----------
    `project`
        W&B project name.
    `wandb_kwargs`
        Extra keyword arguments forwarded to ``wandb.init``.
    """

    def __init__(self, project: str, **wandb_kwargs: Any) -> None:
        try:
            import wandb as _wandb  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "WandbLogger requires the `wandb` package. Install it with:\n  pip install wandb"
            ) from exc
        self._project = project
        self._wandb_kwargs = wandb_kwargs

    def start(self, config: dict[str, Any], run_dir: Path) -> None:
        import wandb

        wandb.init(project=self._project, config=config, dir=str(run_dir), **self._wandb_kwargs)

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        import wandb

        wandb.log(metrics, step=step)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        import wandb

        wandb.config.update(params, allow_val_change=True)

    def finish(self) -> None:
        import wandb

        wandb.finish()
