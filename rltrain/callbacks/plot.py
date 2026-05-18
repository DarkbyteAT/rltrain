"""Plot callback — renders SVG plots of episode returns at each checkpoint.

Reads the rolling buffer of episodes the callback accumulates locally
(matching the JAX-side ``Callback.on_episode_end`` signature) and produces
``per_episode.svg`` and ``per_sample.svg`` next to ``metrics.csv``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import seaborn as sns


log = logging.getLogger(__name__)


class PlotCallback:
    """Renders per-episode and per-sample return SVG plots at each checkpoint.

    Args:
        num_steps: Total number of training steps (x-axis scaling for the
            per-sample plot).
        run_beta: EMA mixing weight used for the running-return curve label.
    """

    def __init__(self, *, num_steps: int, run_beta: float = 0.1) -> None:
        """Initialise the callback with the total step count for x-axis scaling."""
        self._num_steps = num_steps
        self._run_beta = run_beta
        self._run_dir: Path | None = None
        self._returns: list[float] = []
        self._lengths: list[int] = []
        self._running: list[float] = []

    def on_train_start(self, config: dict, run_dir: Path | None) -> None:
        """Record the run directory so plots can be written there."""
        self._run_dir = run_dir

    def on_step(self, step: int, metrics: dict[str, float]) -> None:
        """No-op — plotting is checkpoint-driven, not step-driven."""

    def on_episode_end(
        self,
        episode: int,
        episode_return: float,
        episode_length: int,
        running_return: float = 0.0,
    ) -> None:
        """Buffer one episode's data for the next checkpoint render."""
        self._returns.append(episode_return)
        self._lengths.append(episode_length)
        self._running.append(running_return)

    def on_checkpoint(self, step: int, agent_state, run_dir: Path | None) -> None:
        """Render and save per-episode and per-sample SVG plots."""
        if self._run_dir is None or not self._returns:
            return
        self._plot_episodes()
        self._plot_samples()

    def on_train_end(self, agent_state, run_dir: Path | None) -> None:
        """Render a final pair of plots if not already done."""
        if self._returns:
            self.on_checkpoint(0, agent_state, run_dir)

    def _plot_episodes(self) -> None:
        """Render return-over-episodes SVG."""
        assert self._run_dir is not None
        path = self._run_dir / "per_episode.svg"

        fig = plt.figure(dpi=300, clear=True)
        ax = plt.gca()
        x = np.arange(1, len(self._returns) + 1)
        y_returns = np.asarray(self._returns)
        y_running = np.asarray(self._running)

        plt.title("Return over Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.xlim(1, len(self._returns) + 1)
        ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))

        sns.lineplot(x=x, y=y_returns, color="orange", alpha=0.67, label="Return")
        sns.lineplot(
            x=x,
            y=y_running,
            color="blue",
            label=r"EMA ($\beta = " f"{self._run_beta}" r"$)",
        )
        plt.legend()
        plt.savefig(path, format="svg")
        plt.close(fig)

    def _plot_samples(self) -> None:
        """Render return-over-timesteps SVG."""
        assert self._run_dir is not None
        path = self._run_dir / "per_sample.svg"

        fig = plt.figure(dpi=300, clear=True)
        ax = plt.gca()
        x = np.asarray(self._lengths).cumsum()
        y_returns = np.asarray(self._returns)
        y_running = np.asarray(self._running)

        plt.title("Return over Timesteps")
        plt.xlabel("Timestep")
        plt.ylabel("Return")
        plt.xlim(0, max(self._num_steps, int(x[-1]) if len(x) else 0))
        ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))

        sns.lineplot(x=x, y=y_returns, color="orange", alpha=0.67, label="Return")
        sns.lineplot(
            x=x,
            y=y_running,
            color="blue",
            label=r"EMA ($\beta = " f"{self._run_beta}" r"$)",
        )
        plt.legend()
        plt.savefig(path, format="svg")
        plt.close(fig)
