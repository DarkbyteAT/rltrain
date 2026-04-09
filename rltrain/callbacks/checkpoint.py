"""Checkpoint callback — saves model state_dict at checkpoint intervals."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch as T


if TYPE_CHECKING:
    from rltrain.agents.agent import Agent
    from rltrain.env import MDP


log = logging.getLogger(__name__)


class CheckpointCallback:
    """Saves model ``state_dict`` at each checkpoint and a final model at train end.

    Args:
        save_all: If True, save an intermediate checkpoint at every checkpoint interval.
            If False, only save the final model at train end.
    """

    def __init__(self, *, save_all: bool = False) -> None:
        """Initialise the callback and set the ``save_all`` flag."""
        self.save_all = save_all
        self._models_path: Path | None = None

    def on_train_start(self, agent: Agent, env: MDP, run_dir: Path) -> None:
        """Create the ``models/`` subdirectory inside ``run_dir``."""
        self._models_path = run_dir / "models"
        self._models_path.mkdir(parents=True, exist_ok=True)

    def on_step(self, agent: Agent, env: MDP, step: int) -> None:
        """See ``Callback.on_step``."""
        ...

    def on_episode_end(self, agent: Agent, env: MDP, episode: int) -> None:
        """See ``Callback.on_episode_end``."""
        ...

    def on_checkpoint(self, agent: Agent, env: MDP, run_dir: Path) -> None:
        """Save an intermediate checkpoint when ``save_all`` is enabled."""
        if self.save_all and self._models_path is not None:
            model_path = self._models_path / f"model_{env.episode_steps}.pt"
            T.save(agent.model.state_dict(), model_path)
            log.info("saved checkpoint to '%s'", model_path)

    def on_train_end(self, agent: Agent, env: MDP, run_dir: Path) -> None:
        """Save the final model ``state_dict`` at ``model_FINAL.pt``."""
        if self._models_path is not None:
            model_path = self._models_path / "model_FINAL.pt"
            T.save(agent.model.state_dict(), model_path)
            log.info("saved final model to '%s'", model_path)
