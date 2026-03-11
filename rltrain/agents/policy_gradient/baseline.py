import numpy as np
import torch as T

from rltrain.agents.policy_gradient import VanillaPG
from rltrain.utils import center, discount
from torch.nn.utils import clip_grad_norm_

class REINFORCE(VanillaPG):
    
    name: str = "REINFORCE"
    
    def __init__(self, beta_critic: float, **kwargs):
        super().__init__(**kwargs)
        self.beta_critic = beta_critic
    
    def setup(self):
        super().setup()
        self.critic = self.model["critic"].to(self.device)
        self.critic_opt = self.opt["critic"](self.critic.parameters())
    
    def loss(self, *batch: T.Tensor) -> T.Tensor:
        states, actions, rewards, _, dones = batch
        states = states.float().to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        
        action_dst = self.act(states)
        log_probs = self.log_probs(action_dst, actions)
        entropy = action_dst.entropy().squeeze()
        
        # Whiten returns before computing baseline
        returns = discount(rewards, dones, self.gamma)
        if self.normalise: returns = center(returns)
        baseline = self.critic(states).squeeze()
        advantages = returns - baseline
        
        actor_loss = T.mean(-log_probs * advantages.detach())
        critic_loss = self.beta_critic * T.mean(advantages ** 2)
        entropy_loss = -self.tau * T.mean(entropy)
        return actor_loss + critic_loss + entropy_loss
    
    def descend(self):
        if self.grad_clip is not None:
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        self.actor_opt.step()
        self.critic_opt.step()
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()