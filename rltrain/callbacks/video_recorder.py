"""Video recorder callback — records evaluation videos at checkpoint intervals."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np


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

    def on_train_start(self, agent: Agent, env: MDP, run_dir: Path) -> None:
        self._preprocess_obs = env.preprocess_obs

        try:
            base_env = self._env_fn() if self._env_fn is not None else self._make_env_from_mdp(env)
        except Exception:
            log.warning("VideoRecorderCallback: failed to create eval env — disabling", exc_info=True)
            self._enabled = False
            return

        if base_env.render_mode != "rgb_array":
            log.warning(
                "VideoRecorderCallback: eval env render_mode is '%s', not 'rgb_array' — disabling",
                base_env.render_mode,
            )
            base_env.close()
            self._enabled = False
            return

        video_dir = run_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

        record_kwargs: dict = {
            "video_folder": str(video_dir),
            "episode_trigger": lambda _: True,
            "name_prefix": self._name_prefix,
            "fps": self._fps,
            "disable_logger": True,
        }
        if self._video_length > 0:
            record_kwargs["video_length"] = self._video_length

        self._eval_env = gym.wrappers.RecordVideo(base_env, **record_kwargs)
        log.info("VideoRecorderCallback: recording to '%s'", video_dir)

    def on_step(self, agent: Agent, env: MDP, step: int) -> None: ...

    def on_episode_end(self, agent: Agent, env: MDP, episode: int) -> None:
        if self._episode_trigger is not None and self._episode_trigger(episode):
            self._run_eval_rollouts(agent)

    def on_checkpoint(self, agent: Agent, env: MDP, run_dir: Path) -> None:
        if self._episode_trigger is None:
            self._run_eval_rollouts(agent)

    def on_train_end(self, agent: Agent, env: MDP, run_dir: Path) -> None:
        if self._eval_env is not None:
            self._eval_env.close()
            log.info("VideoRecorderCallback: eval env closed")

    def _make_env_from_mdp(self, env: MDP) -> gym.Env:
        """Auto-detect env ID from the training MDP and create a renderable copy."""
        spec = env.env.envs[0].spec
        if spec is None:
            raise RuntimeError("Cannot auto-detect env — provide env_fn to VideoRecorderCallback")
        return gym.make(spec, render_mode="rgb_array")

    def _run_eval_rollouts(self, agent: Agent) -> None:
        """Run num_episodes evaluation episodes on the recording env."""
        if not self._enabled or self._eval_env is None:
            return
        for _ in range(self._num_episodes):
            obs, _ = self._eval_env.reset()
            terminated, truncated = False, False
            while not (terminated or truncated):
                processed = self._preprocess_obs(obs[np.newaxis, ...])
                action = agent(processed)[0]
                obs, _, terminated, truncated, _ = self._eval_env.step(action)
