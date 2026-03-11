import abc
import logging
import numpy as np
import torch as T
import torch.distributions as dst
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rltrain.env import MDP, Trajectory
from rltrain.utils import get_grad, set_grad
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from typing import Callable, Iterable, Union

class Agent(abc.ABC):
    
    name: str
    log: logging.Logger
    memory: list[Trajectory]
    
    def __init__(self, *,
            name: str = None,
            model: nn.ModuleDict,
            opt: dict[str, Callable[[Iterable[nn.Parameter]], optim.Optimizer]],
            device: T.device,
            gamma: float,
            grad_clip: Union[float, None] = None,
            robust: bool = False,
            rho_sam: float = 1e-2,
            eps_lamp: float = 5e-3,
            rollback_len: int = 0
        ):
        
        if name is not None:
            self.name = name
        elif self.name is None:
            # Hopefully I never see this, something has gone horribly wrong if I do...
            self.name = "AmmarBot"
        
        self.log = logging.getLogger(self.name)
        self.memory = []
        self.model = model
        self.opt = opt
        self.device = device
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.robust = robust
        self.eps_lamp = eps_lamp
        self.rho_sam = rho_sam
        self.rollback_len = rollback_len
        
        if robust:
            self.name += "-LAMP" if rollback_len > 0 else "-SAM"
            self.mean_params = T.zeros_like(parameters_to_vector(self.model.parameters()))
            self.rollback_step = 0
    
    @T.inference_mode()
    def __call__(self, states: np.ndarray) -> np.ndarray:
        action_dst = self.act(T.as_tensor(states).float().to(self.device))
        return action_dst.sample().detach().cpu().numpy()
    
    @abc.abstractmethod
    def setup(self):
        """Sets up the required models and optimisers for the agent to begin training."""
        pass
    
    @abc.abstractmethod
    def act(self, states: T.Tensor) -> dst.Distribution:
        """Returns the distribution representing the policy of the agent.
        
        Parameters
        ----------
        ``states`` : ``Tensor``
            A batch of states to obtain the policy over.
        
        Returns
        -------
        ``Distribution``
            A PyTorch ``Distribution``, which abstracts the implementation of the policy.
        """
        pass
    
    @abc.abstractmethod
    def step(self, env: MDP):
        """Performs a single step in the given environment, using the policy of this agent. This
        also updates the agent's policy if its conditions for performing the update are satisfied.
        
        Parameters
        ----------
        ``env`` : ``MDP``
            The environment to perform the step in.
        """
        pass
    
    @abc.abstractmethod
    def load(self) -> tuple[T.Tensor, ...]:
        """Loads a batch of data from memory, with each datapoint as a batched ``Tensor``.
        
        Returns
        -------
        ``tuple[Tensor, ...]``
            A batch of data to train upon for the given agent, however, not loaded onto the training
            device to avoid memory usage when applying minibatches over training data.
        """
    
    @abc.abstractmethod
    def loss(self, *batch: T.Tensor) -> T.Tensor:
        """Computes the loss function for the policy, over the given batch.
        
        Parameters
        ----------
        ``*batch``
            Batch of experiences to update the agent over.
        
        Returns
        -------
        ``Tensor``
            A single-element tensor containing the mean loss over the given batch.
        """
        pass
    
    @abc.abstractmethod
    def descend(self):
        """Performs a gradient descent step over the parameters of the model, assuming any losses
        have already been backpropagated."""
        pass
    
    def learn(self, *batch: T.Tensor):
        """Updates the policy of the agent with the given batch of data.
        
        Parameters
        ----------
        ``*batch``
            Batch of experiences to update the agent over.
        """
        
        loss = self.loss(*batch)
        loss.backward()
        
        # Guard statement if not using SAM / LAMP (Local-Averaging over Multiple Perturbations)
        if not self.robust:
            self.descend()
            return
        
        # Store original parameters, and compute direction of gradient. Worst case perturbation
        # by moving in direction that increases loss.
        init_params = parameters_to_vector(self.model.parameters())
        init_grad = F.normalize(get_grad(self.model.parameters()), dim=0)
        adv_params = init_params + (self.rho_sam * init_grad)
        vector_to_parameters(adv_params, self.model.parameters())
        loss = self.loss(*batch)
        loss.backward()
        
        # Move back to original parameters, and perform gradient descent with gradient at the
        # worst-case perturbation, but from original parameters.
        new_grad = get_grad(self.model.parameters())
        vector_to_parameters(init_params, self.model.parameters())
        set_grad(new_grad, self.model.parameters())
        self.descend()
        
        # Update moving average + add noise to parameters if using LAMP instead of SAM
        if self.rollback_len > 0:
            # Sample uniform noise, multiply by parameters elementwise to scale
            # Multiply by size of parameters before normalising to adapt to scale of parameters?
            update_params = parameters_to_vector(self.model.parameters())
            noise = update_params.abs() * T.empty_like(init_params).uniform_(-1., 1.)
            noisy_params = update_params + (self.eps_lamp * F.normalize(noise, dim=0))
            vector_to_parameters(noisy_params, self.model.parameters())
            self.mean_params += noisy_params.detach()
            self.rollback_step += 1
            
            # Rollback to moving average if enough updates sampled
            if self.rollback_step > self.rollback_len:
                self.mean_params /= self.rollback_step
                vector_to_parameters(self.mean_params.detach(), self.model.parameters())
                self.mean_params = T.zeros_like(self.mean_params)
                self.rollback_step = 0