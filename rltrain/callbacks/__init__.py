"""Callback protocol and built-in implementations for the training loop."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Callback(Protocol):
    """Five-hook observer protocol for the training loop.

    All hooks receive Python values (not JAX arrays). Callbacks must not
    modify agent or environment state -- they are observers.

    Hooks that are not needed should be implemented as no-ops (pass or ...).
    """

    def on_train_start(self, config: dict, run_dir: Path | None) -> None:
        """Called once before the training loop begins."""
        ...

    def on_step(self, step: int, metrics: dict[str, float]) -> None:
        """Called after each agent.learn() call with Python-float metrics."""
        ...

    def on_episode_end(
        self,
        episode: int,
        episode_return: float,
        episode_length: int,
        running_return: float,
    ) -> None:
        """Called when an episode completes.

        ``running_return`` is the EMA over completed-episode returns,
        computed by the env layer. It is part of the contract — every
        trainer passes it, every callback receives it.
        """
        ...

    def on_checkpoint(self, step: int, agent_state, run_dir: Path | None) -> None:
        """Called at checkpoint intervals (Python-level, outside scan)."""
        ...

    def on_train_end(self, agent_state, run_dir: Path | None) -> None:
        """Called once after the training loop exits."""
        ...


__all__ = ["Callback"]
