import gym.vector as vgym
import logging
import numpy as np

from rltrain.env import Trajectory
from rltrain.utils import lerp
from typing import Callable

class MDP:
    """Simple wrapper around the OpenAI ``gym`` framework that automatically resets environments, as
    well as tracking several statistics, and handling (minimal) input-preprocessing."""
    
    target_reward: float
    
    total_steps: int
    episode_steps: int
    episode_count: int
    run_reward: float
    length_history: list[int]
    return_history: list[float]
    run_history: list[float]
    
    state: np.ndarray
    done: bool
    length: int
    reward_sum: float
    
    def __init__(self, env: vgym.VectorEnv, run_beta: float, log_freq: int, swap_channels: bool):
        self.log = logging.getLogger("Environment")
        self.env = env
        self.run_beta = run_beta
        self.log_freq = log_freq
        self.swap_channels = swap_channels
    
    def setup(self, seed: int):
        self.env.seed(seed)
        self.target_reward = self.env.envs[0].spec.reward_threshold
        self.log.info(f"{self.env.envs[0].observation_space.shape=}")
        self.log.info(f"{self.env.envs[0].action_space=}")
        
        if self.target_reward is not None:
            self.log.info(f"{self.target_reward=:.3f}")
        
        self.total_steps = 0
        self.episode_steps = 0
        self.episode_count = 0
        self.run_reward = None
        self.length_history = []
        self.return_history = []
        self.run_history = []
        
        self.reset()
    
    def reset(self):
        self.state = self.env.reset()
        self.done = False
        self.length = 0
        self.reward_sum = 0.0
        
        # Adjusts next observation to put channels axis first if input is an image
        if self.swap_channels:
            self.state = self.state.squeeze().T[np.newaxis, :]
    
    def step(self, policy: Callable[[np.ndarray], np.ndarray]) -> Trajectory[np.ndarray]:
        """
        Takes a step in the environment, using the given policy function for action-selection.
        Environments are automatically managed by this function, and the following metrics are
        recorded:
        
        - Length (``length_history``)
        - Episode returns (``cum_history``)
        - EMA of returns (``run_history``)
        
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
        next_state, reward, done, _ = self.env.step(action)
        
        # Adjusts next observation to put channels axis first if input is an image
        if self.swap_channels:
            next_state = next_state.squeeze().T[np.newaxis, :]
        
        self.state = next_state
        self.reward_sum += reward[0]
        self.done = done[0]
        self.length += 1
        self.total_steps += 1
        
        # Appends results for completed episode to history-lists for graphing purposes
        if self.done:
            self.run_reward = self.reward_sum if self.run_reward is None else \
                                lerp(self.run_reward, self.reward_sum, self.run_beta)
            self.length_history.append(self.length)
            self.return_history.append(self.reward_sum)
            self.run_history.append(self.run_reward)
            self.episode_steps += self.length
            self.episode_count += 1
            
            if self.episode_count % self.log_freq == 0 or self.run_reward >= self.target_reward:
                self.log.info(
                    f"episode={self.episode_count}\t"
                    f"step={self.total_steps}\t"
                    f"length={self.length}\t"
                    f"return={self.reward_sum:.3f}\t"
                    f"run={self.run_reward:.3f}"
                )
            
            self.reset()
        
        return Trajectory(state, action, reward, next_state, done)