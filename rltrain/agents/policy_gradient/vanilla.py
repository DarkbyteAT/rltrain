"""Vanilla policy gradient with entropy regularisation."""

import time

import numpy as np
import torch as T
import torch.distributions as dst
from torch.nn.utils import clip_grad_norm_

from rltrain.agents import Agent
from rltrain.env import MDP
from rltrain.utils import center, discount


class VanillaPG(Agent):
    """Baseline-free policy gradient agent supporting discrete and continuous action spaces."""

    name: str = "Vanilla Policy-Gradient"

    def __init__(
        self, *, tau: float, eps_per_rollout: int = 1, normalise: bool = False, continuous: bool = False, **kwargs
    ):
        """Initialize with entropy weight, rollout length, and action-space flags."""
        super().__init__(**kwargs)
        self.tau = tau
        self.eps_per_rollout = eps_per_rollout
        self.normalise = normalise
        self.continuous = continuous
        self.policy = self.gaussian_policy if self.continuous else self.softmax_policy

    def setup(self):
        """Initialise the actor network and its optimiser."""
        self.eps_counter = 0
        self.actor = self.model["actor"].to(self.device)
        self.actor_opt = self.opt["actor"](self.actor.parameters())

    def gaussian_policy(self, policy_params: T.Tensor) -> dst.MultivariateNormal:
        """Build a diagonal Gaussian policy from ``(mu, log_sigma)`` interleaved outputs."""
        mu, log_sigma = (policy_params[:, i::2] for i in range(2))
        return dst.MultivariateNormal(mu, log_sigma.exp().diag_embed())

    def softmax_policy(self, policy_params: T.Tensor) -> dst.Categorical:
        """Build a categorical softmax policy from logits."""
        return dst.Categorical(logits=policy_params)

    def log_probs(self, action_dst: dst.Distribution, actions: T.Tensor) -> T.Tensor:
        """Compute log-probabilities for ``actions`` under ``action_dst``."""
        if self.continuous:
            # Check if the action is batched, if not add extra dimension
            if len(actions.shape) == 1:
                actions = actions.unsqueeze(1)
            return action_dst.log_prob(actions).squeeze(-1)
        else:
            return action_dst.log_prob(actions.squeeze(-1)).squeeze(-1)

    def act(self, states: T.Tensor) -> dst.Distribution:
        """Return the policy distribution for a batch of states."""
        return self.policy(self.actor(states))

    def step(self, env: MDP):
        """Collect a transition; run ``learn`` once ``eps_per_rollout`` episodes finish."""
        trajectory = env.step(self)
        self.memory.append(trajectory)

        if trajectory.done.any():
            self.eps_counter += int(trajectory.done.sum())

            if self.eps_counter >= self.eps_per_rollout:
                epoch_time = -time.time()
                self.learn(*self.load())
                epoch_time += time.time()

                self.log.debug(f"{epoch_time=:.3f}s")
                self.memory.clear()

    def load(self) -> tuple[T.Tensor, ...]:
        """Stack memory trajectories into a batched tuple of tensors."""
        # Stack trajectories and merge (timesteps, num_envs) into a single batch dim.
        # Trajectory order: state, action, reward, next_state, done
        arrays = [np.stack(x) for x in zip(*self.memory, strict=False)]
        tensors = [T.from_numpy(a.reshape(-1, *a.shape[2:])) for a in arrays]
        # done (idx 4) stays bool for bitwise ~; actions (idx 1) stay long for
        # Categorical.log_prob in discrete mode, float for continuous.
        result = []
        for i, t in enumerate(tensors):
            if i == 4:
                result.append(t.bool())
            elif i == 1 and not self.continuous:
                result.append(t.long())
            else:
                result.append(t.float())
        return tuple(result)

    def loss(self, *batch: T.Tensor) -> T.Tensor:
        """Compute the REINFORCE-style policy gradient loss with entropy regularisation."""
        states, actions, rewards, _, dones = batch
        states = states.float().to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        # Compute log-probabilities and entropy bonus, with optional return whitening
        action_dst = self.act(states)
        log_probs = self.log_probs(action_dst, actions)
        entropy = action_dst.entropy().squeeze()
        returns = discount(rewards, dones, self.gamma)
        if self.normalise:
            returns = center(returns)

        actor_loss = T.mean(-log_probs * returns)
        entropy_loss = T.mean(-self.tau * entropy)
        return actor_loss + entropy_loss

    def descend(self):
        """Step the actor optimiser, applying gradient clipping if configured."""
        if self.grad_clip is not None:
            clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.actor_opt.step()
        self.actor_opt.zero_grad()
