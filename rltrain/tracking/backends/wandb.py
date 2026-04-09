"""WandbLogger — experiment tracking via Weights & Biases."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class WandbLogger:
    """Wraps ``wandb.init``, ``wandb.log``, ``wandb.config.update``, and ``wandb.finish``.

    Args:
        project: W&B project name.
        wandb_kwargs: Extra keyword arguments forwarded to ``wandb.init``.
    """

    def __init__(self, project: str, **wandb_kwargs: Any) -> None:
        """Validate that wandb is installed and store the project name.

        Args:
            project: W&B project name.
            **wandb_kwargs: Extra keyword arguments forwarded to ``wandb.init``.
        """
        try:
            import wandb as _wandb  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "WandbLogger requires the `wandb` package. Install it with:\n  pip install wandb"
            ) from exc
        self._project = project
        self._wandb_kwargs = wandb_kwargs

    def start(self, config: dict[str, Any], run_dir: Path) -> None:
        """Initialise a W&B run with the experiment config and output directory."""
        import wandb

        wandb.init(project=self._project, config=config, dir=str(run_dir), **self._wandb_kwargs)  # type: ignore[reportAttributeAccessIssue]  # wandb uses dynamic module exports

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to W&B at the given step via ``wandb.log``."""
        import wandb

        wandb.log(metrics, step=step)  # type: ignore[reportAttributeAccessIssue]

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """Update the W&B run config with the given hyperparameters."""
        import wandb

        wandb.config.update(params, allow_val_change=True)  # type: ignore[reportAttributeAccessIssue]

    def finish(self) -> None:
        """Finalise and close the W&B run."""
        import wandb

        wandb.finish()  # type: ignore[reportAttributeAccessIssue]
