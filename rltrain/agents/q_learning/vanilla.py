import numpy as np
import random
import time
import torch as T
import torch.distributions as dst
import torch.nn.functional as F

from copy import deepcopy
from rltrain.agents import Agent
from rltrain.env import MDP
from rltrain.utils import lerp
from torch.nn.utils import clip_grad_norm_, parameters_to_vector, vector_to_parameters

class VanillaDQN(Agent):
    
    name: str = "Vanilla Deep Q-Network"
    
    def __init__(self, *,
            eps_max: float = 1.0,
            eps_min: float = 0.05,
            eps_decay: float = 5e-4,
            memory_size: int = 25000,
            replay_start: int = 10000,
            batch_size: int = 128,
            steps_per_epoch: int = 8,
            target_rate: float = 1e-3,
            **kwargs
        ):
        
        super().__init__(**kwargs)
        self.eps_greedy = eps_max
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.memory_size = memory_size
        self.replay_start = replay_start
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.target_rate = target_rate
    
    def setup(self):
        self.step_counter = 0
        self.qnet = self.model["qnet"].to(self.device)
        self.qnet_opt = self.opt["qnet"](self.qnet.parameters())
        self.target = deepcopy(self.model["qnet"])
    
    def act(self, states: T.Tensor) -> dst.Distribution:
        q_values = self.qnet(states)
        opt_actions = q_values.argmax(dim=1).long()
        action_space = q_values.shape[1]
        
        # Best action = True, otherwise = False
        # p(Best): 1 - eps_greedy
        # p(Other): eps_greedy / (|A| - 1)
        return dst.Categorical(probs=
            T.where(F.one_hot(opt_actions, action_space).bool(),
                1 - self.eps_greedy,
                self.eps_greedy / (action_space - 1)
            )
        )
    
    def step(self, env: MDP):
        trajectory = env.step(self)
        self.memory.append(trajectory)
        self.step_counter += 1
        
        if len(self.memory) >= self.replay_start:
            # Remove oldest experience if replay memory too large
            if len(self.memory) > self.memory_size:
                del self.memory[0]
            
            if self.step_counter % self.steps_per_epoch == 0:
                epoch_time = -time.time()
                self.learn(*self.load())
                epoch_time += time.time()
                self.log.debug(f"{epoch_time=:.3f}s")
    
    def load(self) -> tuple[T.Tensor, ...]:
        # Sample random batch from replay memory
        batch = random.sample(self.memory, k=self.batch_size)
        return tuple(T.from_numpy(np.asarray(x).squeeze()) for x in zip(*batch))
    
    def loss(self, *batch: T.Tensor) -> T.Tensor:
        states, actions, rewards, next_states, dones = batch
        states = states.float().to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.float().to(self.device)
        dones = dones.to(self.device)
        
        # Compute MSE loss with target-network maximum
        q_values = self.qnet(states).gather(1, actions.unsqueeze(1)).squeeze()
        q_values_target = self.target(next_states).amax(dim=1).squeeze()
        q_max = rewards + (~dones * self.gamma * q_values_target)
        return T.mean((q_max - q_values) ** 2)
    
    def descend(self):
        if self.grad_clip is not None:
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        self.qnet_opt.step()
        self.qnet_opt.zero_grad()
        
        # Decrease epsilon to increase greedy actions
        self.eps_greedy = max(self.eps_min, self.eps_greedy - self.eps_decay)
        # Soft-update for target network parameters
        target_params = parameters_to_vector(self.target.parameters())
        online_params = parameters_to_vector(self.qnet.parameters())
        lerp_params = lerp(target_params, online_params, self.target_rate)
        vector_to_parameters(lerp_params.detach(), self.target.parameters())