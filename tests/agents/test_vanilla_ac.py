"""Hand-computed loss test for VanillaAC.

Loss = mean(-log_π(a|s) · δ.detach()) + β · mean(δ²) + (-τ · mean(H(π)))

where δ = r + γ·V(s') - V(s) is the one-step TD error. The TD error is detached
for the actor loss (psi) but not for the critic loss, so the critic gradient
flows through the value predictions.
"""

import torch as T

from rltrain.agents.actor_critic import VanillaAC
from tests.agents.conftest import make_ac_agent


# Pre-computed with tests/agents/_compute_expected.py (seed actor=0, critic=1)
EXPECTED_LOSS = 0.22458896040916443


def test_vanilla_ac_loss(batch_5):
    agent = make_ac_agent(VanillaAC)
    loss = agent.loss(*batch_5)
    assert T.allclose(loss, T.tensor(EXPECTED_LOSS), atol=1e-6), (
        f"VanillaAC loss mismatch: got {loss.item()}, expected {EXPECTED_LOSS}"
    )
