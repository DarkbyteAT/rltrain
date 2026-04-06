import logging
from collections.abc import Callable

import gymnasium.vector as vgym
import numpy as np

from rltrain.env.trajectory import Trajectory
from rltrain.utils import lerp


class MDP:
    """Simple wrapper around the ``gymnasium`` framework that automatically resets environments, as
    well as tracking several statistics, and handling (minimal) input-preprocessing.

    Supports both single-env (``num_envs=1``) and multi-env vectorised operation.
    ``SyncVectorEnv`` auto-resets done environments, so the MDP only calls ``reset()``
    once at setup time.  Per-env episode tracking (length, return, running return) is
    maintained independently via arrays.
    """

    target_reward: float | None

    total_steps: int
    episode_steps: int
    episode_count: int
    run_reward: float | None
    length_history: list[int]
    return_history: list[float]
    run_history: list[float]

    state: np.ndarray
    _lengths: np.ndarray
    _reward_sums: np.ndarray

    def __init__(self, env: vgym.VectorEnv, run_beta: float, log_freq: int, swap_channels: bool):
        self.log = logging.getLogger("Environment")
        self.env = env
        self.num_envs: int = env.num_envs
        self.run_beta = run_beta
        self.log_freq = log_freq
        self.swap_channels = swap_channels

    def preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        """Apply observation preprocessing (e.g. channel swap for image envs).

        Parameters
        ----------
        `obs` : `np.ndarray`
            Raw observation from the gymnasium environment.

        Returns
        -------
        `np.ndarray`
            Preprocessed observation ready for the agent.
        """
        if self.swap_channels:
            return np.transpose(obs, (0, 3, 1, 2))
        return obs

    def setup(self, seed: int | None):
        self.seed = seed
        self.target_reward = self.env.envs[0].spec.reward_threshold  # type: ignore[reportAttributeAccessIssue]  # gymnasium stub gap: SyncVectorEnv.envs exists at runtime
        self.log.info(f"{self.env.envs[0].observation_space.shape=}")  # type: ignore[reportAttributeAccessIssue]
        self.log.info(f"{self.env.envs[0].action_space=}")  # type: ignore[reportAttributeAccessIssue]
        self.log.info(f"{self.num_envs=}")

        if self.target_reward is not None:
            self.log.info(f"{self.target_reward=:.3f}")

        self.total_steps = 0
        self.episode_steps = 0
        self.episode_count = 0
        self.run_reward = None
        self.length_history = []
        self.return_history = []
        self.run_history = []

        self._lengths = np.zeros(self.num_envs, dtype=np.int64)
        self._reward_sums = np.zeros(self.num_envs, dtype=np.float64)

        self.state, _ = self.env.reset(seed=seed)
        self.state = self.preprocess_obs(self.state)

    def step(self, policy: Callable[[np.ndarray], np.ndarray]) -> Trajectory[np.ndarray]:
        """Takes a step in the environment using the given policy function for action-selection.

        ``SyncVectorEnv`` automatically resets sub-environments that are done, so
        ``next_state`` for a finished environment is already the first observation of
        the *new* episode.  Episode statistics are recorded per-env before the internal
        counters are zeroed.

        Parameters
        ----------
        ``policy`` : ``(ndarray) -> ndarray``
            Policy function for the agent(s), i.e. mapping states to actions.

        Returns
        -------
        ``Trajectory[ndarray]``
            ``(state, action, reward, next_state, done)``
        """
        state = self.state
        action = policy(state)
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated | truncated

        next_state = self.preprocess_obs(next_state)

        self.state = next_state
        self._lengths += 1
        self._reward_sums += reward
        self.total_steps += self.num_envs

        # Record completed episodes — may be zero, one, or many per step.
        done_mask = done.astype(bool)

        if done_mask.any():
            for i in np.where(done_mask)[0]:
                ep_length = int(self._lengths[i])
                ep_return = float(self._reward_sums[i])

                self.run_reward = (
                    ep_return if self.run_reward is None else lerp(self.run_reward, ep_return, self.run_beta)
                )
                self.length_history.append(ep_length)
                self.return_history.append(ep_return)
                self.run_history.append(self.run_reward)
                self.episode_steps += ep_length
                self.episode_count += 1

                if self.episode_count % self.log_freq == 0 or (
                    self.target_reward is not None
                    and self.run_reward is not None
                    and self.run_reward >= self.target_reward
                ):
                    self.log.info(
                        f"episode={self.episode_count}\t"
                        f"step={self.total_steps}\t"
                        f"length={ep_length}\t"
                        f"return={ep_return:.3f}\t"
                        f"run={self.run_reward:.3f}"
                    )

            # Reset per-env counters for finished envs only.
            # SyncVectorEnv has already auto-reset the underlying env.
            self._lengths[done_mask] = 0
            self._reward_sums[done_mask] = 0.0

        return Trajectory(state, action, reward, next_state, done)
