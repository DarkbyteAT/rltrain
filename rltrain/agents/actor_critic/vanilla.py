import torch as T
import torch.nn as nn

from rltrain.agents.policy_gradient import REINFORCE
from rltrain.utils import center


class VanillaAC(REINFORCE):
    name: str = "Vanilla Actor-Critic"

    def __init__(self, *, shared_features: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.shared_features = shared_features

    def setup(self):
        if self.shared_features:
            self.embedding = self.model["embedding"].to(self.device)
            self.actor = nn.Sequential(self.embedding, self.model["actor"].to(self.device))
            self.critic = nn.Sequential(self.embedding, self.model["critic"].to(self.device))
            self.actor_opt = self.opt["actor"](self.actor.parameters())
            self.critic_opt = self.opt["critic"](self.critic.parameters())
        else:
            super().setup()

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

        # Whiten TD error, but only use to create better advantage for actor to avoid bias
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        deltas = rewards + (~dones * self.gamma * next_values) - values
        psi = (center(deltas) if self.normalise else deltas).detach().clone()

        actor_loss = T.mean(-log_probs * psi)
        critic_loss = self.beta_critic * T.mean(deltas**2)
        entropy_loss = -self.tau * T.mean(entropy)
        return actor_loss + critic_loss + entropy_loss
