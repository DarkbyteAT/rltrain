"""Callback protocol and built-in callbacks for the training loop."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:
    from rltrain.agents.agent import Agent
    from rltrain.env import MDP


@runtime_checkable
class Callback(Protocol):
    """Hook points for the training loop. All methods have default no-ops."""

    def on_train_start(self, agent: Agent, env: MDP, run_dir: Path) -> None:
        """Called once before training begins; use for setup and initialisation."""
        ...

    def on_step(self, agent: Agent, env: MDP, step: int) -> None:
        """Called after every ``agent.step()`` call with the current step count."""
        ...

    def on_episode_end(self, agent: Agent, env: MDP, episode: int) -> None:
        """Called when an episode completes with the current episode count."""
        ...

    def on_checkpoint(self, agent: Agent, env: MDP, run_dir: Path) -> None:
        """Called at each checkpoint interval; use for saving metrics or models."""
        ...

    def on_train_end(self, agent: Agent, env: MDP, run_dir: Path) -> None:
        """Called once after the training loop exits; use for cleanup and finalisation."""
        ...
