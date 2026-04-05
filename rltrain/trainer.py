"""Trainer — owns the training loop and callback orchestration."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch as T

from rltrain.agents.agent import Agent
from rltrain.callbacks import Callback
from rltrain.callbacks.checkpoint import CheckpointCallback
from rltrain.callbacks.csv_logger import CSVLoggerCallback
from rltrain.callbacks.plot import PlotCallback
from rltrain.env import MDP


log = logging.getLogger(__name__)


class Trainer:
    """Runs the training loop with callback hooks.

    The Trainer owns ONLY the training loop and callback orchestration.
    It does NOT own arg parsing, config loading, or object instantiation.

    Parameters
    ----------
    `agent` : `Agent`
        The RL agent to train.
    `env` : `MDP`
        The environment to train in.
    `num_steps` : `int`
        Total number of environment steps to train for.
    `checkpoint_steps` : `int`
        Number of steps between checkpoint callbacks.
    `run_dir` : `Path`
        Directory for saving outputs (models, metrics, plots).
    `callbacks` : `list[Callback] | None`
        List of callbacks. Defaults to CSV + Plot + Checkpoint if None.
    `seed` : `int | None`
        RNG seed for reproducibility. If None, no seeding is performed.
    """

    def __init__(
        self,
        agent: Agent,
        env: MDP,
        *,
        num_steps: int,
        checkpoint_steps: int,
        run_dir: Path,
        callbacks: list[Callback] | None = None,
        seed: int | None = None,
    ) -> None:
        self.agent = agent
        self.env = env
        self.num_steps = num_steps
        self.checkpoint_steps = checkpoint_steps
        self.run_dir = run_dir
        self.seed = seed

        if callbacks is None:
            self.callbacks: list[Callback] = [
                CSVLoggerCallback(),
                PlotCallback(num_steps=num_steps),
                CheckpointCallback(),
            ]
        else:
            self.callbacks = callbacks

    def fit(self) -> None:
        """Run the training loop.

        Calls ``agent.setup()`` and ``env.setup(seed)`` if `seed` is provided,
        then runs the agent-environment loop with callback hooks at each stage.
        """
        if self.seed is not None:
            log.info("seeding global RNGs (seed=%d)...", self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)
            T.manual_seed(self.seed)

        self.agent.setup()
        self.env.setup(self.seed)

        for cb in self.callbacks:
            cb.on_train_start(self.agent, self.env, self.run_dir)

        last_checkpoint = 0
        last_episode = 0

        while self.env.total_steps < self.num_steps:
            self.agent.step(self.env)

            for cb in self.callbacks:
                cb.on_step(self.agent, self.env, self.env.total_steps)

            # Fire on_episode_end once per completed episode. With multi-env,
            # several sub-envs can finish in one step, so the counter may jump
            # by more than 1.
            while self.env.episode_count > last_episode:
                last_episode += 1
                for cb in self.callbacks:
                    cb.on_episode_end(self.agent, self.env, last_episode)

            if self.env.total_steps != last_checkpoint and (
                self.env.total_steps % self.checkpoint_steps == 0 or self.env.total_steps >= self.num_steps
            ):
                last_checkpoint = self.env.total_steps
                for cb in self.callbacks:
                    cb.on_checkpoint(self.agent, self.env, self.run_dir)

        for cb in self.callbacks:
            cb.on_train_end(self.agent, self.env, self.run_dir)
