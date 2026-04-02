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

    Parameters
    ----------
    `save_all` : `bool`
        If True, save an intermediate checkpoint at every checkpoint interval.
        If False, only save the final model at train end.
    """

    def __init__(self, *, save_all: bool = False) -> None:
        self.save_all = save_all
        self._models_path: Path | None = None

    def on_train_start(self, agent: Agent, env: MDP, run_dir: Path) -> None:
        self._models_path = run_dir / "models"
        self._models_path.mkdir(parents=True, exist_ok=True)

    def on_step(self, agent: Agent, env: MDP, step: int) -> None: ...
    def on_episode_end(self, agent: Agent, env: MDP, episode: int) -> None: ...

    def on_checkpoint(self, agent: Agent, env: MDP, run_dir: Path) -> None:
        if self.save_all and self._models_path is not None:
            model_path = self._models_path / f"model_{env.episode_steps}.pt"
            T.save(agent.model.state_dict(), model_path)
            log.info("saved checkpoint to '%s'", model_path)

    def on_train_end(self, agent: Agent, env: MDP, run_dir: Path) -> None:
        if self._models_path is not None:
            model_path = self._models_path / "model_FINAL.pt"
            T.save(agent.model.state_dict(), model_path)
            log.info("saved final model to '%s'", model_path)
