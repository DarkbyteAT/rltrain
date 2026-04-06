"""Hand-computed loss test for VanillaDQN.

Loss = mean((r + γ · max_a' Q_target(s', a') - Q(s, a))²)

DQN uses a separate target network (initialised as a copy of the online network)
for computing TD targets, and a replay buffer for decorrelating samples.
"""

import functools

import pytest
import torch as T
import torch.nn as nn
import torch.optim as optim

from rltrain.agents.q_learning import VanillaDQN
from tests.agents.conftest import GAMMA


# Pre-computed with tests/agents/_compute_expected.py (seed qnet=0, target=0 copy)
EXPECTED_LOSS = 0.8412540555000305


@pytest.mark.unit
def test_dqn_loss(batch_5):
    T.manual_seed(0)
    qnet = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2))
    model = nn.ModuleDict({"qnet": qnet})
    agent = VanillaDQN(
        model=model,
        opt={"qnet": functools.partial(optim.Adam, lr=1e-3)},
        device=T.device("cpu"),
        gamma=GAMMA,
    )
    agent.setup()

    loss = agent.loss(*batch_5)
    assert T.allclose(loss, T.tensor(EXPECTED_LOSS), atol=1e-6), (
        f"DQN loss mismatch: got {loss.item()}, expected {EXPECTED_LOSS}"
    )
