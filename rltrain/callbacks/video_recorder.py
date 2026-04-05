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


def _identity(obs: np.ndarray) -> np.ndarray:
    return obs


class VideoRecorderCallback:
    """Records evaluation videos of agent behaviour during training.

    Runs evaluation rollouts on a separate environment and writes MP4 files
    named by training step (e.g. ``step-25000.mp4``). By default, records at
    each checkpoint; optionally configure ``eval_trigger`` to record at
    specific training episodes instead.

    Parameters
    ----------
    `env_fn` : `Callable[[], gym.Env] | None`
        Zero-arg callable returning a ``gym.Env`` with ``render_mode="rgb_array"``.
        If None, auto-detects from the training MDP's env spec. The auto-detection
        creates a bare env without user-applied wrappers; pass ``env_fn`` explicitly
        when wrappers matter for the recording.
    `num_episodes` : `int`
        Number of evaluation episodes to record at each trigger point.
    `eval_trigger` : `Callable[[int], bool] | None`
        When set, controls when eval rollouts happen during training based on the
        training episode count. Rollouts trigger in ``on_episode_end`` instead of
        the default ``on_checkpoint``. For example, ``lambda ep: ep % 50 == 0``
        records every 50th training episode.
    `video_length` : `int`
        Maximum video length in frames per episode. 0 means record full episodes.
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
        eval_trigger: Callable[[int], bool] | None = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        fps: int = 30,
    ) -> None:
        self._env_fn = env_fn
        self._num_episodes = num_episodes
        self._eval_trigger = eval_trigger
        self._video_length = video_length
        self._name_prefix = name_prefix
        self._fps = fps

        self._eval_env: gym.Env | None = None
        self._preprocess_obs: Callable[[np.ndarray], np.ndarray] = _identity
        self._video_dir: Path | None = None
        self._enabled: bool = True

    def on_train_start(self, agent: Agent, env: MDP, run_dir: Path) -> None:
        self._preprocess_obs = env.preprocess_obs

        try:
            self._eval_env = self._env_fn() if self._env_fn is not None else self._make_env_from_mdp(env)
        except Exception:
            log.warning("VideoRecorderCallback: failed to create eval env — disabling", exc_info=True)
            self._enabled = False
            return

        if self._eval_env.render_mode != "rgb_array":
            log.warning(
                "VideoRecorderCallback: eval env render_mode is '%s', not 'rgb_array' — disabling",
                self._eval_env.render_mode,
            )
            self._eval_env.close()
            self._eval_env = None
            self._enabled = False
            return

        self._video_dir = run_dir / "videos"
        self._video_dir.mkdir(parents=True, exist_ok=True)
        log.info("VideoRecorderCallback: recording to '%s'", self._video_dir)

    def on_step(self, agent: Agent, env: MDP, step: int) -> None: ...

    def on_episode_end(self, agent: Agent, env: MDP, episode: int) -> None:
        if self._eval_trigger is not None and self._eval_trigger(episode):
            self._run_eval_rollouts(agent, env.total_steps)

    def on_checkpoint(self, agent: Agent, env: MDP, run_dir: Path) -> None:
        if self._eval_trigger is None:
            self._run_eval_rollouts(agent, env.total_steps)

    def on_train_end(self, agent: Agent, env: MDP, run_dir: Path) -> None:
        if self._eval_env is not None:
            self._eval_env.close()
            log.info("VideoRecorderCallback: eval env closed")

    def _make_env_from_mdp(self, env: MDP) -> gym.Env:
        """Auto-detect env from the training MDP and create a renderable copy.

        Passes the full ``EnvSpec`` to preserve any custom kwargs from the
        original environment registration.
        """
        spec = env.env.envs[0].spec
        if spec is None:
            raise RuntimeError("Cannot auto-detect env — provide env_fn to VideoRecorderCallback")
        return gym.make(spec, render_mode="rgb_array")

    def _run_eval_rollouts(self, agent: Agent, step: int) -> None:
        """Run evaluation episodes and save each as a step-named MP4."""
        if not self._enabled or self._eval_env is None or self._video_dir is None:
            return

        import moviepy

        for ep in range(self._num_episodes):
            frames: list[np.ndarray] = []
            obs, _ = self._eval_env.reset()
            terminated, truncated = False, False

            while not (terminated or truncated):
                frame = self._eval_env.render()
                if frame is not None:
                    frames.append(frame)
                if 0 < self._video_length <= len(frames):
                    break
                processed = self._preprocess_obs(obs[np.newaxis, ...])
                action = agent(processed)[0]
                obs, _, terminated, truncated, _ = self._eval_env.step(action)

            if not frames:
                log.warning("VideoRecorderCallback: no frames captured at step %d", step)
                continue

            suffix = f"-{ep}" if self._num_episodes > 1 else ""
            path = self._video_dir / f"{self._name_prefix}-step-{step}{suffix}.mp4"
            clip = moviepy.ImageSequenceClip(frames, fps=self._fps)
            clip.write_videofile(str(path), logger=None)
            clip.close()
            log.debug("VideoRecorderCallback: wrote '%s' (%d frames)", path, len(frames))
