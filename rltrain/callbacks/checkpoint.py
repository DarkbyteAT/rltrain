"""Checkpoint callback — serialises agent state at checkpoint intervals."""

from __future__ import annotations

import logging
from pathlib import Path

import equinox as eqx


log = logging.getLogger(__name__)


class CheckpointCallback:
    """Saves agent state via ``eqx.tree_serialise_leaves`` at each checkpoint.

    Files land under ``run_dir/models/`` named ``model_{step}.eqx``, with a
    final ``model_FINAL.eqx`` written at train end. To resume training, use
    ``eqx.tree_deserialise_leaves(path, agent_state_template)``.

    Args:
        save_all: When True, save an intermediate checkpoint at every interval.
            When False, only the final model is saved at train end.
    """

    def __init__(self, *, save_all: bool = False) -> None:
        """Initialise the callback and the ``save_all`` flag."""
        self.save_all = save_all
        self._models_path: Path | None = None

    def on_train_start(self, config: dict, run_dir: Path | None) -> None:
        """Create the ``models/`` subdirectory inside ``run_dir``."""
        if run_dir is not None:
            self._models_path = Path(run_dir) / "models"
            self._models_path.mkdir(parents=True, exist_ok=True)

    def on_step(self, step: int, metrics: dict[str, float]) -> None:
        """No-op — checkpointing is interval-driven."""

    def on_episode_end(
        self,
        episode: int,
        episode_return: float,
        episode_length: int,
        running_return: float = 0.0,
    ) -> None:
        """No-op — checkpointing is interval-driven."""

    def on_checkpoint(self, step: int, agent_state, run_dir: Path | None) -> None:
        """Serialise an intermediate checkpoint when ``save_all`` is enabled."""
        if self.save_all and self._models_path is not None:
            path = self._models_path / f"model_{step}.eqx"
            eqx.tree_serialise_leaves(str(path), agent_state)
            log.info("saved checkpoint to '%s'", path)

    def on_train_end(self, agent_state, run_dir: Path | None) -> None:
        """Serialise the final agent state to ``model_FINAL.eqx``."""
        if self._models_path is not None:
            path = self._models_path / "model_FINAL.eqx"
            eqx.tree_serialise_leaves(str(path), agent_state)
            log.info("saved final model to '%s'", path)
