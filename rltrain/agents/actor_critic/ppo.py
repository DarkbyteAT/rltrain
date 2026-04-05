import random
import time

import numpy as np
import torch as T
from torch.distributions import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from rltrain.agents.actor_critic import AdvantageAC
from rltrain.env import MDP
from rltrain.utils import center, discount


class PPO(AdvantageAC):
    name: str = "Proximal Policy Optimisation"

    def __init__(self, *, num_epochs: int, batch_size: int, early_stop: float, eps_clip: float, **kwargs):
        super().__init__(**kwargs)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.eps_clip = eps_clip

    def step(self, env: MDP):
        trajectory = env.step(self)
        self.memory.append(trajectory)

        if len(self.memory) >= self.horizon:
            epoch_time = -time.time()
            backtrack_params = parameters_to_vector(self.model.parameters()).detach()
            dataset = self.load()
            batch_idx = np.arange(0, self.horizon, self.batch_size)
            stop = False

            for _ in range(self.num_epochs):
                # Shuffle starting indices for mini-batches
                random.shuffle(batch_idx)

                for i in batch_idx:
                    self.learn(*[x[i : i + self.batch_size] for x in dataset])
                    # Check if the algorithm should stop early
                    stop = self.check_kl(dataset[0], dataset[-3])

                    # Update backtrack parameters if not stopping, else set back to last checkpoint
                    # and break to prevent further training
                    if not stop:
                        backtrack_params = parameters_to_vector(self.model.parameters()).detach()
                    else:
                        vector_to_parameters(backtrack_params.detach(), self.model.parameters())
                        break

                # Break from outer loop also
                if stop:
                    break

            epoch_time += time.time()
            self.log.debug(f"{epoch_time=:.3f}s")
            self.memory.clear()

    def check_kl(self, states: T.Tensor, policy_old: T.Tensor) -> bool:
        """Returns ``True`` if the updated policy's KL-divergence exceeds ``early_stop``.

        Parameters
        ----------
        ``states`` : ``Tensor``
            States over which the mean KL-divergence is to be compared over.
        ``policy_old`` : ``Tensor``
            Output layer of the original policy's actor network at the given states.
        """

        new_dst = self.act(states)
        old_dst = self.policy(policy_old)
        return bool(kl_divergence(old_dst, new_dst).mean() >= self.early_stop)

    def load(self) -> tuple[T.Tensor, ...]:
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
