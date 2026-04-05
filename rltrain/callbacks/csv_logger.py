"""CSV logger callback — writes episode metrics to CSV at checkpoint intervals."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd


if TYPE_CHECKING:
    from rltrain.agents.agent import Agent
    from rltrain.env import MDP


log = logging.getLogger(__name__)


class CSVLoggerCallback:
    """Writes episode metrics (episode, length, return, running_return) to CSV.

    The CSV is written at each checkpoint and at train end to capture the full
    episode history up to that point.
    """

    def __init__(self) -> None:
        self._csv_path: Path | None = None

    def on_train_start(self, agent: Agent, env: MDP, run_dir: Path) -> None:
        self._csv_path = run_dir / "metrics.csv"

    def on_step(self, agent: Agent, env: MDP, step: int) -> None: ...
    def on_episode_end(self, agent: Agent, env: MDP, episode: int) -> None: ...

    def on_checkpoint(self, agent: Agent, env: MDP, run_dir: Path) -> None:
        self._write(env)

    def on_train_end(self, agent: Agent, env: MDP, run_dir: Path) -> None:
        self._write(env)

    def _write(self, env: MDP) -> None:
        if self._csv_path is None or env.episode_count == 0:
            return
        pd.DataFrame(
            {
                "episode": np.arange(1, env.episode_count + 1),
                "length": np.asarray(env.length_history),
                "return": np.asarray(env.return_history),
                "running_return": np.asarray(env.run_history),
            }
        ).set_index("episode").to_csv(self._csv_path)
        log.debug("wrote metrics to '%s'", self._csv_path)
