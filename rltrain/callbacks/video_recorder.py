"""Video recorder callback — records evaluation videos at checkpoint intervals."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym


if TYPE_CHECKING:
    from rltrain.agents.agent import Agent
    from rltrain.env import MDP


log = logging.getLogger(__name__)


class VideoRecorderCallback:
    """Records evaluation videos of agent behaviour during training.

    Uses gymnasium's ``RecordVideo`` wrapper on a separate persistent evaluation
    environment. By default, records at each checkpoint; optionally configure
    ``episode_trigger`` to record at specific training episodes instead.

    Parameters
    ----------
    `env_fn` : `Callable[[], gym.Env] | None`
        Zero-arg callable returning a ``gym.Env`` with ``render_mode="rgb_array"``.
        If None, auto-detects from the training MDP's env spec. The auto-detection
        creates a bare env without user-applied wrappers; pass ``env_fn`` explicitly
        when wrappers matter for the recording.
    `num_episodes` : `int`
        Number of evaluation episodes to record at each trigger point.
    `episode_trigger` : `Callable[[int], bool] | None`
        When set, controls when eval rollouts happen during training based on the
        training episode count. Rollouts trigger in ``on_episode_end`` instead of
        the default ``on_checkpoint``.
    `video_length` : `int`
        Fixed video length in frames. 0 means record full episodes.
    `name_prefix` : `str`
        Filename prefix for recorded videos.
    `fps` : `int`
        Frames per second for the output video.
    """

    def __init__(
        self,
        *,
        env_fn: Callable[[], gym.Env] | None = None,
        num_episodes: int = 3,
        episode_trigger: Callable[[int], bool] | None = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        fps: int = 30,
    ) -> None:
        self._env_fn = env_fn
        self._num_episodes = num_episodes
        self._episode_trigger = episode_trigger
        self._video_length = video_length
        self._name_prefix = name_prefix
        self._fps = fps

        self._eval_env: gym.Env | None = None
        self._enabled: bool = True

    def on_train_start(self, agent: Agent, env: MDP, run_dir: Path) -> None: ...
    def on_step(self, agent: Agent, env: MDP, step: int) -> None: ...
    def on_episode_end(self, agent: Agent, env: MDP, episode: int) -> None: ...
    def on_checkpoint(self, agent: Agent, env: MDP, run_dir: Path) -> None: ...
    def on_train_end(self, agent: Agent, env: MDP, run_dir: Path) -> None: ...
