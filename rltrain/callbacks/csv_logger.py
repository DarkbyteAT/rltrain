"""CSV logger callback -- writes episode metrics to a CSV file."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import IO


class CSVLoggerCallback:
    """Writes episode metrics (return, length, EMA running return) to a CSV file.

    Accumulates episode data between checkpoints, then flushes to disk.
    """

    def __init__(self) -> None:
        """Initialise empty episode buffer and file handles."""
        self._episodes: list[dict] = []
        self._writer: csv.writer | None = None
        self._file: IO[str] | None = None

    def on_train_start(self, config: dict, run_dir: Path | None) -> None:
        """Open metrics.csv in run_dir and write the header row."""
        if run_dir:
            path = Path(run_dir) / "metrics.csv"
            self._file = open(path, "w", newline="")  # noqa: SIM115
            self._writer = csv.writer(self._file)
            self._writer.writerow(["episode", "return", "length", "running_return"])

    def on_step(self, step: int, metrics: dict[str, float]) -> None:
        """No-op -- step-level metrics are not logged to CSV."""

    def on_episode_end(
        self,
        episode: int,
        episode_return: float,
        episode_length: int,
        running_return: float = 0.0,
    ) -> None:
        """Buffer episode data for the next checkpoint flush."""
        self._episodes.append(
            {
                "episode": episode,
                "return": episode_return,
                "length": episode_length,
                "running_return": running_return,
            }
        )

    def on_checkpoint(self, step: int, agent_state, run_dir: Path | None) -> None:
        """Flush buffered episodes to the CSV file."""
        if self._writer:
            for ep in self._episodes:
                self._writer.writerow([ep["episode"], ep["return"], ep["length"], ep["running_return"]])
            assert self._file is not None
            self._file.flush()
            self._episodes.clear()

    def on_train_end(self, agent_state, run_dir: Path | None) -> None:
        """Flush remaining episodes and close the CSV file."""
        self.on_checkpoint(0, agent_state, run_dir)
        if self._file:
            self._file.close()
