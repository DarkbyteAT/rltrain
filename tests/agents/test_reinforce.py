"""Hand-computed loss test for REINFORCE.

Loss = mean(-log_π(a|s) · (G_t - V(s)).detach()) + β · mean((G_t - V(s))²) + (-τ · mean(H(π)))

REINFORCE subtracts a learned baseline V(s) from the discounted returns to reduce
variance. The advantage (G_t - V(s)) is detached for the actor loss so the critic
gradient doesn't flow through the policy gradient term.
"""

import pytest
import torch as T

from rltrain.agents.policy_gradient import REINFORCE
from tests.agents.conftest import make_baseline_agent


# Pre-computed with tests/agents/_compute_expected.py (seed actor=0, critic=1)
EXPECTED_LOSS = -0.18018627166748047


@pytest.mark.unit
def test_reinforce_loss(batch_5):
    agent = make_baseline_agent(REINFORCE)
    loss = agent.loss(*batch_5)
    assert T.allclose(loss, T.tensor(EXPECTED_LOSS), atol=1e-6), (
        f"REINFORCE loss mismatch: got {loss.item()}, expected {EXPECTED_LOSS}"
    )
