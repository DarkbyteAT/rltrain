"""Abstract base ``Agent`` — template-method orchestration for RL algorithms."""

import abc
import logging
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import torch as T
import torch.distributions as dst
import torch.nn as nn
import torch.optim as optim
from samgria import GradientTransform

from rltrain.env import MDP, Trajectory


class Agent(abc.ABC):
    """Abstract RL agent — subclasses implement ``setup``, ``act``, ``step``, ``load``, ``loss``, ``descend``."""

    name: str
    log: logging.Logger
    memory: list[Trajectory]

    def __init__(
        self,
        *,
        name: str | None = None,
        model: nn.ModuleDict,
        opt: dict[str, Callable[[Iterable[nn.Parameter]], optim.Optimizer]],
        device: T.device,
        gamma: float,
        grad_clip: float | None = None,
        grad_transforms: Sequence[GradientTransform] = (),
    ):
        """Initialize the agent with model, optimiser factories, and hyperparameters."""
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
        self.grad_transforms = grad_transforms
        for transform in self.grad_transforms:
            self.name += f"-{type(transform).__name__}"

    @T.inference_mode()
    def __call__(self, states: np.ndarray) -> np.ndarray:
        """Sample actions from the agent's policy for a batch of states."""
        action_dst = self.act(T.as_tensor(states).float().to(self.device))
        return action_dst.sample().detach().cpu().numpy()

    @abc.abstractmethod
    def setup(self):
        """Sets up the required models and optimisers for the agent to begin training."""
        pass

    @abc.abstractmethod
    def act(self, states: T.Tensor) -> dst.Distribution:
        """Returns the distribution representing the policy of the agent.

        Args:
            states: A batch of states to obtain the policy over.

        Returns:
            A PyTorch ``Distribution``, which abstracts the implementation of the policy.
        """
        pass

    @abc.abstractmethod
    def step(self, env: MDP):
        """Perform a single step in the environment and update the policy when conditions allow.

        This also updates the agent's policy if its conditions for performing
        the update are satisfied.

        Args:
            env: The environment to perform the step in.
        """
        pass

    @abc.abstractmethod
    def load(self) -> tuple[T.Tensor, ...]:
        """Loads a batch of data from memory, with each datapoint as a batched ``Tensor``.

        Returns:
            A batch of data to train upon for the given agent, however, not loaded onto the training
            device to avoid memory usage when applying minibatches over training data.
        """

    @abc.abstractmethod
    def loss(self, *batch: T.Tensor) -> T.Tensor:
        """Computes the loss function for the policy, over the given batch.

        Args:
            *batch: Batch of experiences to update the agent over.

        Returns:
            A single-element tensor containing the mean loss over the given batch.
        """
        pass

    @abc.abstractmethod
    def descend(self):
        """Perform a gradient descent step over the model parameters.

        Assumes any losses have already been backpropagated.
        """
        pass

    def learn(self, *batch: T.Tensor):
        """Updates the policy of the agent with the given batch of data.

        The optimisation pipeline is:

        1. Compute loss and backpropagate.
        2. Apply each ``GradientTransform.apply()`` in order (pre-descent).
        3. Call ``descend()`` (optimiser step).
        4. Apply each ``GradientTransform.post_step()`` in order (post-descent).

        Args:
            *batch: Batch of experiences to update the agent over.
        """
        loss = self.loss(*batch)
        loss.backward()

        for transform in self.grad_transforms:
            transform.apply(self.model, self.loss, batch)

        self.descend()

        for transform in self.grad_transforms:
            transform.post_step(self.model)
