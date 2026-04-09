"""Proximal Policy Optimisation (PPO) — clipped surrogate with mini-batch epochs."""

import time
from collections.abc import Sequence

import numpy as np
import torch as T
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from rltrain.agents.actor_critic.a2c import AdvantageAC
from rltrain.agents.actor_critic.ppo.epoch_terminator import EpochTerminator
from rltrain.env import MDP
from rltrain.utils import center, discount


class PPO(AdvantageAC):
    """PPO agent — clipped surrogate objective with mini-batch epochs and KL early stop."""

    name: str = "Proximal Policy Optimisation"

    def __init__(
        self,
        *,
        num_epochs: int,
        batch_size: int,
        eps_clip: float,
        epoch_terminators: Sequence[EpochTerminator] = (),
        **kwargs,
    ):
        """Initialize PPO with mini-batch epoch, clipping, and terminator configuration."""
        super().__init__(**kwargs)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.eps_clip = eps_clip
        self.epoch_terminators = epoch_terminators

    def step(self, env: MDP):
        """Collect a transition, run mini-batch epochs with KL backtracking when the horizon fills."""
        trajectory = env.step(self)
        self.memory.append(trajectory)

        if len(self.memory) >= self.horizon:
            epoch_time = -time.time()
            dataset = self.load()
            batch_idx = np.arange(0, len(dataset[0]), self.batch_size)
            pre_epoch_params = parameters_to_vector(self.model.parameters()).detach()

            for _ in range(self.num_epochs):
                np.random.shuffle(batch_idx)
                approx_kl = 0.0

                for i in batch_idx:
                    mini_batch = [x[i : i + self.batch_size] for x in dataset]
                    self.learn(*mini_batch)

                    # Approximate KL from log ratios — cheap, no extra forward pass
                    with T.no_grad():
                        approx_kl = self._approx_kl(mini_batch)

                # Check epoch terminators
                triggered = [t for t in self.epoch_terminators if t.should_stop(approx_kl)]
                if triggered:
                    if any(t.rollback for t in triggered):
                        vector_to_parameters(pre_epoch_params, self.model.parameters())
                    break

                pre_epoch_params = parameters_to_vector(self.model.parameters()).detach()

            epoch_time += time.time()
            self.log.debug(f"{epoch_time=:.3f}s")
            self.memory.clear()

    def _approx_kl(self, mini_batch: list[T.Tensor]) -> float:
        """Compute approximate KL divergence from the last mini-batch's log ratios.

        Uses the improved estimator from Schulman's blog:
        ``mean((ratio - 1) - log(ratio))``, which is always non-negative.

        Args:
            mini_batch: The standard PPO batch tuple ``(states, actions, rewards,
                next_states, dones, policy_old, advantages, returns)``. Only
                ``states``, ``actions``, and ``policy_old`` are used here.

        Returns:
            Scalar approximate KL divergence between the current and old policy.
        """
        states, actions, _r, _ns, _d, policy_old, _adv, _ret = mini_batch
        action_dst = self.act(states)
        old_dst = self.policy(policy_old)
        log_ratio = self.log_probs(action_dst, actions) - self.log_probs(old_dst, actions)
        ratio = log_ratio.exp()
        return float(((ratio - 1) - log_ratio).mean())

    def load(self) -> tuple[T.Tensor, ...]:
        """Return the standard batch plus pre-computed policy outputs, advantages, and returns."""
        states, actions, rewards, next_states, dones = super().load()

        # Compute advantages before epochs, gives value function a stationary target
        with T.no_grad():
            states = states.float().to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.float().to(self.device)
            dones = dones.to(self.device)

            policy_old = self.actor(states)
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            deltas = rewards + (self.gamma * ~dones * next_values) - values
            advantages = discount(deltas, dones, self.gamma * self.lambda_gae)
            returns = (advantages + values).detach().clone()
            if self.normalise:
                advantages = center(advantages).detach().clone()

        return states, actions, rewards, next_states, dones, policy_old, advantages, returns

    def loss(self, *batch: T.Tensor) -> T.Tensor:
        """Compute the clipped PPO surrogate loss with critic and entropy terms."""
        states, actions, rewards, next_states, dones, policy_old, advantages, returns = batch
        states = states.float().to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.float().to(self.device)
        dones = dones.to(self.device)
        policy_old = policy_old.detach().clone().squeeze()
        advantages = advantages.detach().clone().squeeze()
        returns = returns.detach().clone().squeeze()

        action_dst = self.act(states)
        old_dst = self.policy(policy_old)
        log_probs = self.log_probs(action_dst, actions)
        log_probs_old = self.log_probs(old_dst, actions).detach().clone()
        entropy = action_dst.entropy().squeeze()
        values = self.critic(states).squeeze()

        # e^(lnp - lnq) = e^(ln(p/q)) = p/q
        imp_ratio = (log_probs - log_probs_old).exp()
        true_ratio = imp_ratio * advantages
        clip_ratio = T.clamp(imp_ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

        # Take minimum of ratios as in PPO, prevents loss from exploding due to ratio
        actor_loss = -T.min(true_ratio, clip_ratio).mean()
        critic_loss = self.beta_critic * T.mean((returns - values) ** 2)
        entropy_loss = -self.tau * T.mean(entropy)
        return actor_loss + critic_loss + entropy_loss
