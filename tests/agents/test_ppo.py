"""Hand-computed loss test for PPO.

Loss = -mean(min(r·A, clip(r, 1-ε, 1+ε)·A)) + β · mean((R - V(s))²) + (-τ · mean(H(π)))

PPO's loss() receives 8 tensors (states, actions, rewards, next_states, dones,
policy_old, advantages, returns) — the last three are pre-computed by load().

When policy_old = current policy output (first epoch, no weight update), the
importance ratio r = π_new(a|s) / π_old(a|s) = 1, so clipping has no effect
and the loss equals the standard A2C policy gradient.
"""

import torch as T

from rltrain.agents.actor_critic import PPO
from rltrain.utils import discount
from tests.agents.conftest import EPS_CLIP, GAMMA, LAMBDA_GAE, make_ac_agent


# Pre-computed with tests/agents/_compute_expected.py (seed actor=0, critic=1)
EXPECTED_LOSS = 1.00101900100708


def test_ppo_loss(batch_5):
    agent = make_ac_agent(
        PPO,
        horizon=128,
        lambda_gae=LAMBDA_GAE,
        num_epochs=4,
        batch_size=32,
        early_stop=0.2,
        eps_clip=EPS_CLIP,
    )

    states, actions, rewards, next_states, dones = batch_5

    # Pre-compute advantages and returns the same way PPO.load() does
    with T.no_grad():
        policy_old = agent.actor(states.float())
        values = agent.critic(states.float()).squeeze()
        next_values = agent.critic(next_states.float()).squeeze()
        deltas = rewards + (GAMMA * ~dones * next_values) - values
        advantages = discount(deltas, dones, GAMMA * LAMBDA_GAE)
        returns = (advantages + values).detach().clone()

    batch_8 = (states, actions, rewards, next_states, dones, policy_old, advantages, returns)
    loss = agent.loss(*batch_8)
    assert T.allclose(loss, T.tensor(EXPECTED_LOSS), atol=1e-6), (
        f"PPO loss mismatch: got {loss.item()}, expected {EXPECTED_LOSS}"
    )
