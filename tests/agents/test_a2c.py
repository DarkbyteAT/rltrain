"""Hand-computed loss test for AdvantageAC (A2C).

Loss = mean(-log_π(a|s) · A^GAE) + β · mean((R_target - V(s))²) + (-τ · mean(H(π)))

where:
  δ_t = r_t + γ·V(s_{t+1}).detach() - V(s_t)     (TD errors)
  A^GAE_t = discount(δ.detach(), dones, γ·λ)       (GAE advantages)
  R_target = A^GAE + V(s)                           (critic target)
"""

import torch as T

from rltrain.agents.actor_critic import AdvantageAC
from tests.agents.conftest import LAMBDA_GAE, make_ac_agent


# Pre-computed with tests/agents/_compute_expected.py (seed actor=0, critic=1)
EXPECTED_LOSS = -0.16368556022644043


def test_a2c_loss(batch_5):
    agent = make_ac_agent(AdvantageAC, horizon=128, lambda_gae=LAMBDA_GAE)
    loss = agent.loss(*batch_5)
    assert T.allclose(loss, T.tensor(EXPECTED_LOSS), atol=1e-6), (
        f"A2C loss mismatch: got {loss.item()}, expected {EXPECTED_LOSS}"
    )
