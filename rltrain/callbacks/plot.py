"""Plot callback — renders per-episode and per-sample SVG plots at checkpoints."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import seaborn as sns

from rltrain.agents.agent import Agent
from rltrain.env import MDP


log = logging.getLogger(__name__)


class PlotCallback:
    """Renders per-episode and per-sample return SVG plots at each checkpoint.

    Parameters
    ----------
    `num_steps` : `int`
        Total number of training steps (used for x-axis scaling on the
        per-sample plot).
    """

    def __init__(self, *, num_steps: int) -> None:
        self._num_steps = num_steps
        self._run_dir: Path | None = None

    def on_train_start(self, agent: Agent, env: MDP, run_dir: Path) -> None:
        self._run_dir = run_dir

    def on_step(self, agent: Agent, env: MDP, step: int) -> None: ...
    def on_episode_end(self, agent: Agent, env: MDP, episode: int) -> None: ...

    def on_checkpoint(self, agent: Agent, env: MDP, run_dir: Path) -> None:
        if self._run_dir is None or env.episode_count == 0:
            return
        self._plot_episodes(agent, env)
        self._plot_samples(agent, env)

    def on_train_end(self, agent: Agent, env: MDP, run_dir: Path) -> None: ...

    def _plot_episodes(self, agent: Agent, env: MDP) -> None:
        """Render return-over-episodes SVG."""
        assert self._run_dir is not None
        episode_plot_path = self._run_dir / "per_episode.svg"

        log.debug("plotting episode graph...")
        fig = plt.figure(dpi=600, clear=True)
        ax = plt.gca()
        x = np.arange(1, env.episode_count + 1)
        y1 = np.asarray(env.return_history)
        y2 = np.asarray(env.run_history)

        plt.title(f"Return over Episodes ({agent.name})")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.xlim(1, env.episode_count + 1)
        ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))

        if env.target_reward is not None:
            ax.set_ylim(
                min(env.target_reward, y1.min(), y2.min()),
                max(env.target_reward, y1.max(), y2.max()),
            )
            plt.axhline(env.target_reward, linestyle="dashed", color="black", label="Target")

        sns.lineplot(x=x, y=y1, color="orange", alpha=0.67, label="Return")
        sns.lineplot(
            x=x,
            y=y2,
            color="blue",
            label=r"EMA ($\beta = " f"{env.run_beta}" r"$)",
        )
        plt.legend()
        plt.plot()
        plt.savefig(episode_plot_path, format="svg")
        plt.close(fig)

    def _plot_samples(self, agent: Agent, env: MDP) -> None:
        """Render return-over-timesteps SVG."""
        assert self._run_dir is not None
        sample_plot_path = self._run_dir / "per_sample.svg"

        log.debug("plotting per-sample graph...")
        fig = plt.figure(dpi=600, clear=True)
        ax = plt.gca()
        x = np.asarray(env.length_history).cumsum()
        y1 = np.asarray(env.return_history)
        y2 = np.asarray(env.run_history)

        plt.title(f"Return over Timesteps ({agent.name})")
        plt.xlabel("Timestep")
        plt.ylabel("Return")
        plt.xlim(0, max(self._num_steps, env.episode_steps))
        ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))

        if env.target_reward is not None:
            ax.set_ylim(
                min(env.target_reward, y1.min(), y2.min()),
                max(env.target_reward, y1.max(), y2.max()),
            )
            plt.axhline(env.target_reward, linestyle="dashed", color="black", label="Target")

        sns.lineplot(x=x, y=y1, color="orange", alpha=0.67, label="Return")
        sns.lineplot(
            x=x,
            y=y2,
            color="blue",
            label=r"EMA ($\beta = " f"{env.run_beta}" r"$)",
        )
        plt.legend()
        plt.plot()
        plt.savefig(sample_plot_path, format="svg")
        plt.close(fig)
