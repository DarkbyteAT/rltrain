"""Video recorder callback -- creates directory structure for evaluation videos."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path


logger = logging.getLogger(__name__)


class VideoRecorderCallback:
    """Records evaluation rollouts as MP4 at checkpoint boundaries.

    Uses a separate gymnasium env for rendering. For the spike, this is a stub
    that creates the directory structure but does not require moviepy.

    Args:
        env_fn: Factory returning a gymnasium env with render_mode="rgb_array".
            If None, video recording is disabled (with a warning).
        num_episodes: Number of evaluation episodes per checkpoint.
        video_dir: Subdirectory under run_dir for videos. Defaults to "videos".
    """

    def __init__(
        self,
        env_fn: Callable | None = None,
        num_episodes: int = 3,
        video_dir: str = "videos",
    ) -> None:
        """Initialise with an optional env factory and recording parameters."""
        self._env_fn = env_fn
        self._num_episodes = num_episodes
        self._video_dir = video_dir
        self._enabled = env_fn is not None

    def on_train_start(self, config: dict, run_dir: Path | None) -> None:
        """No-op -- setup is deferred to checkpoint time."""

    def on_step(self, step: int, metrics: dict[str, float]) -> None:
        """No-op -- video recording is checkpoint-driven."""

    def on_episode_end(self, episode: int, episode_return: float, episode_length: int) -> None:
        """No-op -- episode metrics are not used for video recording."""

    def on_checkpoint(self, step: int, agent_state, run_dir: Path | None) -> None:
        """Create the video directory (stub -- no actual recording in the spike)."""
        if run_dir is None:
            return
        video_path = Path(run_dir) / self._video_dir
        video_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            "VideoRecorderCallback: created %s (step=%d, stub -- no actual recording)",
            video_path,
            step,
        )

    def on_train_end(self, agent_state, run_dir: Path | None) -> None:
        """No-op -- nothing to clean up in the stub."""
