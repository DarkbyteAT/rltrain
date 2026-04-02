"""Fixtures for agent loss function tests.

Provides tiny seeded networks and known batches for hand-computed loss verification.
"""

import functools

import pytest
import torch as T
import torch.nn as nn
import torch.optim as optim

from rltrain.transforms import SAM, LAMPRollback


GAMMA = 0.99
TAU = 0.01
BETA_CRITIC = 0.5
LAMBDA_GAE = 0.95
EPS_CLIP = 0.2


def _make_actor(seed: int = 0) -> nn.Sequential:
    T.manual_seed(seed)
    return nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2))


def _make_critic(seed: int = 1) -> nn.Sequential:
    T.manual_seed(seed)
    return nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))


def _make_qnet(seed: int = 0) -> nn.Sequential:
    T.manual_seed(seed)
    return nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2))


@pytest.fixture
def batch_5():
    """Standard 5-tensor batch: states, actions, rewards, next_states, dones."""
    return (
        T.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        T.tensor([0, 1, 0]),
        T.tensor([1.0, 0.0, -1.0]),
        T.tensor([[0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]),
        T.tensor([False, False, True]),
    )


def _adam_factory(params):
    return optim.Adam(params, lr=1e-3)


def _build_transforms(sam: bool, rollback_len: int):
    """Build a gradient transform pipeline matching the old robust/rollback_len API."""
    transforms = []
    if sam:
        transforms.append(SAM(rho=1e-2))
    if rollback_len > 0:
        transforms.append(LAMPRollback(eps=5e-3, rollback_len=rollback_len))
    return transforms


def make_pg_agent(agent_cls, *, sam=False, rollback_len=0, grad_transforms=None, **extra):
    """Build a tiny policy gradient agent with known weights."""
    actor = _make_actor(0)
    model = nn.ModuleDict({"actor": actor})
    transforms = grad_transforms if grad_transforms is not None else _build_transforms(sam, rollback_len)
    agent = agent_cls(
        model=model,
        opt={"actor": functools.partial(optim.Adam, lr=1e-3)},
        device=T.device("cpu"),
        gamma=GAMMA,
        tau=TAU,
        normalise=False,
        continuous=False,
        grad_transforms=transforms,
        **extra,
    )
    agent.setup()
    return agent


def make_baseline_agent(agent_cls, *, sam=False, rollback_len=0, **extra):
    """Build a tiny REINFORCE agent (actor + critic, no shared_features param)."""
    actor = _make_actor(0)
    critic = _make_critic(1)
    model = nn.ModuleDict({"actor": actor, "critic": critic})
    agent = agent_cls(
        model=model,
        opt={
            "actor": functools.partial(optim.Adam, lr=1e-3),
            "critic": functools.partial(optim.Adam, lr=1e-3),
        },
        device=T.device("cpu"),
        gamma=GAMMA,
        tau=TAU,
        beta_critic=BETA_CRITIC,
        normalise=False,
        continuous=False,
        grad_transforms=_build_transforms(sam, rollback_len),
        **extra,
    )
    agent.setup()
    return agent


def make_ac_agent(agent_cls, *, sam=False, rollback_len=0, **extra):
    """Build a tiny actor-critic agent with known weights."""
    actor = _make_actor(0)
    critic = _make_critic(1)
    model = nn.ModuleDict({"actor": actor, "critic": critic})
    agent = agent_cls(
        model=model,
        opt={
            "actor": functools.partial(optim.Adam, lr=1e-3),
            "critic": functools.partial(optim.Adam, lr=1e-3),
        },
        device=T.device("cpu"),
        gamma=GAMMA,
        tau=TAU,
        beta_critic=BETA_CRITIC,
        normalise=False,
        continuous=False,
        shared_features=False,
        grad_transforms=_build_transforms(sam, rollback_len),
        **extra,
    )
    agent.setup()
    return agent
