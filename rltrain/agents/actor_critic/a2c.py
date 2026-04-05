import time

import torch as T

from rltrain.agents.actor_critic import VanillaAC
from rltrain.env import MDP
from rltrain.utils import center, discount


class AdvantageAC(VanillaAC):
    name: str = "Advantage Actor-Critic"

    def __init__(self, *, horizon: int, lambda_gae: float, **kwargs):
        super().__init__(**kwargs)
        self.horizon = horizon
        self.lambda_gae = lambda_gae

    def step(self, env: MDP):
        trajectory = env.step(self)
        self.memory.append(trajectory)

        if len(self.memory) >= self.horizon:
            epoch_time = -time.time()
            self.learn(*self.load())
            epoch_time += time.time()

            self.log.debug(f"{epoch_time=:.3f}s")
            self.memory.clear()

    def loss(self, *batch: T.Tensor) -> T.Tensor:
        states, actions, rewards, next_states, dones = batch
        states = states.float().to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.float().to(self.device)
        dones = dones.to(self.device)

        action_dst = self.act(states)
        log_probs = self.log_probs(action_dst, actions)
        entropy = action_dst.entropy().squeeze()

        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        deltas = rewards + (~dones * self.gamma * next_values.detach()) - values
        # GAE is a discounted sum of TD errors
        advantages = discount(deltas.detach(), dones, self.gamma * self.lambda_gae)
        returns = (advantages + values).detach().clone()
        if self.normalise:
            advantages = center(advantages)

        actor_loss = T.mean(-log_probs * advantages)
        critic_loss = self.beta_critic * T.mean((returns - values) ** 2)
        entropy_loss = -self.tau * T.mean(entropy)
        return actor_loss + critic_loss + entropy_loss
