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

    def on_train_start(self, agent: Agent, env: MDP, run_dir: Path) -> None: ...
    def on_step(self, agent: Agent, env: MDP, step: int) -> None: ...
    def on_episode_end(self, agent: Agent, env: MDP, episode: int) -> None: ...
    def on_checkpoint(self, agent: Agent, env: MDP, run_dir: Path) -> None: ...
    def on_train_end(self, agent: Agent, env: MDP, run_dir: Path) -> None: ...
