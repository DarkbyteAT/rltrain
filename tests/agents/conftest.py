"""Fixtures for agent loss function tests.

Provides tiny seeded networks and known batches for hand-computed loss verification.
"""

import functools

import pytest
import torch as T
import torch.nn as nn
import torch.optim as optim


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


def make_pg_agent(agent_cls, *, robust=False, rollback_len=0, **extra):
    """Build a tiny policy gradient agent with known weights."""
    actor = _make_actor(0)
    model = nn.ModuleDict({"actor": actor})
    agent = agent_cls(
        model=model,
        opt={"actor": functools.partial(optim.Adam, lr=1e-3)},
        device=T.device("cpu"),
        gamma=GAMMA,
        tau=TAU,
        normalise=False,
        continuous=False,
        robust=robust,
        rollback_len=rollback_len,
        **extra,
    )
    agent.setup()
    return agent


def make_baseline_agent(agent_cls, *, robust=False, rollback_len=0, **extra):
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
        robust=robust,
        rollback_len=rollback_len,
        **extra,
    )
    agent.setup()
    return agent


def make_ac_agent(agent_cls, *, robust=False, rollback_len=0, **extra):
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
        robust=robust,
        rollback_len=rollback_len,
        **extra,
    )
    agent.setup()
    return agent
