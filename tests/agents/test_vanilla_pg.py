"""Hand-computed loss test for VanillaPG.

Loss = mean(-log_π(a|s) · G_t) + mean(-τ · H(π))

where G_t = discount(rewards, dones, γ) is the discounted return.
"""

import pytest
import torch as T

from rltrain.agents.policy_gradient import VanillaPG
from tests.agents.conftest import make_pg_agent


# Pre-computed with tests/agents/_compute_expected.py (seed=0, gamma=0.99, tau=0.01)
EXPECTED_LOSS = -0.47168490290641785


@pytest.mark.unit
def test_vanilla_pg_loss(batch_5):
    agent = make_pg_agent(VanillaPG)
    loss = agent.loss(*batch_5)
    assert T.allclose(loss, T.tensor(EXPECTED_LOSS), atol=1e-6), (
        f"VanillaPG loss mismatch: got {loss.item()}, expected {EXPECTED_LOSS}"
    )
