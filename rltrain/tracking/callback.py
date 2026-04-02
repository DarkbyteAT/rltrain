"""TrackingCallback — adapts the Callback protocol to a MetricsLogger backend."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from rltrain.tracking.logger import MetricsLogger


if TYPE_CHECKING:
    from rltrain.agents.agent import Agent
    from rltrain.env import MDP


class TrackingCallback:
    """Bridges the training loop's ``Callback`` hooks to a ``MetricsLogger``.

    The experiment ``config`` is captured at construction time because
    ``Callback.on_train_start`` receives ``(agent, env, run_dir)`` but not the
    raw configuration dictionary.

    Parameters
    ----------
    `logger`
        Any object satisfying the ``MetricsLogger`` protocol.
    `config`
        Full experiment configuration dictionary.
    """

    def __init__(self, logger: MetricsLogger, config: dict[str, Any]) -> None:
        self._logger = logger
        self._config = config

    # -- Callback hooks -------------------------------------------------------

    def on_train_start(self, agent: Agent, env: MDP, run_dir: Path) -> None:
        self._logger.start(self._config, run_dir)
        self._logger.log_hyperparams(self._config)

    def on_step(self, agent: Agent, env: MDP, step: int) -> None: ...

    def on_episode_end(self, agent: Agent, env: MDP, episode: int) -> None:
        self._logger.log_scalars(
            {
                "return": env.return_history[-1],
                "length": env.length_history[-1],
                "running_return": env.run_history[-1],
            },
            step=episode,
        )

    def on_checkpoint(self, agent: Agent, env: MDP, run_dir: Path) -> None: ...

    def on_train_end(self, agent: Agent, env: MDP, run_dir: Path) -> None:
        self._logger.finish()
