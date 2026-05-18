"""Video recorder callback — records evaluation rollouts as MP4 at checkpoints.

Uses a separate gymnasium env with ``render_mode="rgb_array"`` for rendering.
The agent's ``act()`` is called in eager mode during eval rollouts because
gymnasium envs are opaque Python objects that cannot be traced by JAX.

Requires ``moviepy`` for video writing (``pip install moviepy``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


logger = logging.getLogger(__name__)


class VideoRecorderCallback:
    r"""Records evaluation videos of agent behaviour at checkpoint boundaries.

    At each checkpoint, runs ``num_episodes`` greedy eval rollouts on a
    separate gymnasium env, captures frames via ``env.render()``, and
    writes them as MP4 files using moviepy.

    The agent is captured at construction time — the callback calls
    ``agent.act(state, obs, key)`` during eval rollouts using the
    ``agent_state`` received at each checkpoint.

    Args:
        agent: The agent module (static ``eqx.Module``).  Its ``act()``
            method is called with the checkpoint's ``agent_state``.
        env_fn: Zero-arg callable returning a ``gymnasium.Env`` with
            ``render_mode="rgb_array"``.
        num_episodes: Number of evaluation episodes per checkpoint.
        video_dir: Subdirectory under run_dir for videos.
        max_steps: Maximum steps per eval episode (safety cap).
        fps: Frames per second for the output video.
    """

    def __init__(
        self,
        agent=None,
        env_fn: Callable | None = None,
        num_episodes: int = 3,
        video_dir: str = "videos",
        max_steps: int = 1000,
        fps: int = 30,
    ) -> None:
        """Initialise with agent and env factory."""
        self._agent = agent
        self._env_fn = env_fn
        self._num_episodes = num_episodes
        self._video_dir_name = video_dir
        self._max_steps = max_steps
        self._fps = fps
        self._video_dir: Path | None = None

    def on_train_start(self, config: dict, run_dir: Path | None) -> None:
        """Create the video output directory."""
        if run_dir is not None:
            self._video_dir = Path(run_dir) / self._video_dir_name
            self._video_dir.mkdir(parents=True, exist_ok=True)

    def on_step(self, step: int, metrics: dict[str, float]) -> None:
        """No-op."""

    def on_episode_end(
        self,
        episode: int,
        episode_return: float,
        episode_length: int,
        running_return: float = 0.0,
    ) -> None:
        """No-op."""

    def on_checkpoint(self, step: int, agent_state, run_dir: Path | None) -> None:
        """Record eval rollouts and write MP4 videos."""
        if self._video_dir is None:
            return

        if self._agent is None or self._env_fn is None:
            return

        self._record_rollouts(step, agent_state)

    def on_train_end(self, agent_state, run_dir: Path | None) -> None:
        """No-op."""

    def _record_rollouts(self, step: int, agent_state) -> None:
        """Run eval episodes, capture frames, write MP4."""
        import moviepy

        eval_env = self._env_fn()

        if getattr(eval_env, "render_mode", None) != "rgb_array":
            logger.warning(
                "VideoRecorderCallback: render_mode is '%s', not 'rgb_array' — skipping",
                getattr(eval_env, "render_mode", None),
            )
            eval_env.close()
            return

        for ep in range(self._num_episodes):
            frames: list[np.ndarray] = []
            obs, _info = eval_env.reset()
            terminated, truncated = False, False
            key = jax.random.PRNGKey(step * 1000 + ep)

            for _t in range(self._max_steps):
                frame = eval_env.render()
                if frame is not None:
                    frames.append(np.asarray(frame))

                if terminated or truncated:
                    break

                # Agent.act expects JAX arrays
                obs_jax = jnp.array(obs, dtype=jnp.float32)
                key, k_act = jax.random.split(key)
                action_jax = self._agent.act(agent_state, obs_jax, k_act)

                # Gymnasium expects numpy
                action_np = np.asarray(action_jax)
                obs, _reward, terminated, truncated, _info = eval_env.step(action_np)

            if not frames:
                continue

            suffix = f"-{ep}" if self._num_episodes > 1 else ""
            path = self._video_dir / f"step-{step}{suffix}.mp4"
            clip = moviepy.ImageSequenceClip(frames, fps=self._fps)
            clip.write_videofile(str(path), logger=None)
            clip.close()
            logger.info("Wrote %s (%d frames)", path, len(frames))

        eval_env.close()
