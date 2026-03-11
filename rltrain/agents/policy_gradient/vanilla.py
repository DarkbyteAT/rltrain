import numpy as np
import time
import torch as T
import torch.distributions as dst

from rltrain.agents import Agent
from rltrain.env import MDP
from rltrain.utils import center, discount
from torch.nn.utils import clip_grad_norm_

class VanillaPG(Agent):
    
    name: str = "Vanilla Policy-Gradient"
    
    def __init__(self, *,
            tau: float,
            eps_per_rollout: int = 1,
            normalise: bool = False,
            continuous: bool = False,
            **kwargs
        ):
        
        super().__init__(**kwargs)
        self.tau = tau
        self.eps_per_rollout = eps_per_rollout
        self.normalise = normalise
        self.continuous = continuous
        self.policy = self.gaussian_policy if self.continuous else self.softmax_policy
    
    def setup(self):
        self.eps_counter = 0
        self.actor = self.model["actor"].to(self.device)
        self.actor_opt = self.opt["actor"](self.actor.parameters())
    
    def gaussian_policy(self, policy_params: T.Tensor) -> dst.MultivariateNormal:
        mu, log_sigma = (policy_params[:, i::2] for i in range(2))
        return dst.MultivariateNormal(mu, log_sigma.exp().diag_embed())
    
    def softmax_policy(self, policy_params: T.Tensor) -> dst.Categorical:
        return dst.Categorical(logits=policy_params)
    
    def log_probs(self, action_dst: dst.Distribution, actions: T.Tensor) -> T.Tensor:
        if self.continuous:
            # Check if the action is batched, if not add extra dimension
            if len(actions.shape) == 1: actions.unsqueeze_(1)
            return action_dst.log_prob(actions).squeeze()
        else:
            return action_dst.log_prob(actions.squeeze()).squeeze()
    
    def act(self, states: T.Tensor) -> dst.Distribution:
        return self.policy(self.actor(states))
    
    def step(self, env: MDP):
        trajectory = env.step(self)
        self.memory.append(trajectory)
        
        if trajectory.done[0]:
            self.eps_counter += 1
            
            if self.eps_counter >= self.eps_per_rollout:
                epoch_time = -time.time()
                self.learn(*self.load())
                epoch_time += time.time()
                
                self.log.debug(f"{epoch_time=:.3f}s")
                self.memory.clear()
    
    def load(self) -> tuple[T.Tensor, ...]:
        return tuple(T.from_numpy(np.asarray(x).squeeze()) for x in zip(*self.memory))
    
    def loss(self, *batch: T.Tensor) -> T.Tensor:
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
        if self.normalise: returns = center(returns)
        
        actor_loss = T.mean(-log_probs * returns)
        entropy_loss = T.mean(-self.tau * entropy)
        return actor_loss + entropy_loss
    
    def descend(self):
        if self.grad_clip is not None:
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        self.actor_opt.step()
        self.actor_opt.zero_grad()