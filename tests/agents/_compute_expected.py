"""Helper script to compute expected loss values for agent tests.

Run with: uv run python tests/agents/_compute_expected.py
"""

import torch as T
import torch.distributions as dst
import torch.nn as nn

from rltrain.utils import discount


def main():
    states = T.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    actions = T.tensor([0, 1, 0])
    rewards = T.tensor([1.0, 0.0, -1.0])
    next_states = T.tensor([[0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
    dones = T.tensor([False, False, True])
    gamma = 0.99
    tau = 0.01
    beta_critic = 0.5
    lambda_gae = 0.95
    eps_clip = 0.2

    # VanillaPG
    T.manual_seed(0)
    a = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2))
    logits = a(states)
    d = dst.Categorical(logits=logits)
    lp = d.log_prob(actions)
    ent = d.entropy()
    ret = discount(rewards, dones, gamma)
    al = T.mean(-lp * ret)
    el = T.mean(-tau * ent)
    print(f"VPG_LOSS = {(al + el).item()!r}")

    # REINFORCE
    T.manual_seed(0)
    a2 = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2))
    T.manual_seed(1)
    c2 = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
    logits2 = a2(states)
    d2 = dst.Categorical(logits=logits2)
    lp2 = d2.log_prob(actions)
    ent2 = d2.entropy()
    ret2 = discount(rewards, dones, gamma)
    bl = c2(states).squeeze()
    adv = ret2 - bl
    al2 = T.mean(-lp2 * adv.detach())
    cl2 = beta_critic * T.mean(adv**2)
    el2 = -tau * T.mean(ent2)
    print(f"RF_LOSS = {(al2 + cl2 + el2).item()!r}")

    # VanillaAC
    T.manual_seed(0)
    a3 = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2))
    T.manual_seed(1)
    c3 = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
    logits3 = a3(states)
    d3 = dst.Categorical(logits=logits3)
    lp3 = d3.log_prob(actions)
    ent3 = d3.entropy()
    v3 = c3(states).squeeze()
    nv3 = c3(next_states).squeeze()
    delta3 = rewards + (~dones * gamma * nv3) - v3
    psi3 = delta3.detach().clone()
    al3 = T.mean(-lp3 * psi3)
    cl3 = beta_critic * T.mean(delta3**2)
    el3 = -tau * T.mean(ent3)
    print(f"VAC_LOSS = {(al3 + cl3 + el3).item()!r}")

    # A2C
    T.manual_seed(0)
    a4 = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2))
    T.manual_seed(1)
    c4 = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
    logits4 = a4(states)
    d4 = dst.Categorical(logits=logits4)
    lp4 = d4.log_prob(actions)
    ent4 = d4.entropy()
    v4 = c4(states).squeeze()
    nv4 = c4(next_states).squeeze()
    delta4 = rewards + (~dones * gamma * nv4.detach()) - v4
    adv4 = discount(delta4.detach(), dones, gamma * lambda_gae)
    ret4 = (adv4 + v4).detach().clone()
    al4 = T.mean(-lp4 * adv4)
    cl4 = beta_critic * T.mean((ret4 - v4) ** 2)
    el4 = -tau * T.mean(ent4)
    print(f"A2C_LOSS = {(al4 + cl4 + el4).item()!r}")

    # DQN
    T.manual_seed(0)
    q = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2))
    T.manual_seed(0)
    tgt = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2))
    qv = q(states).gather(1, actions.unsqueeze(1)).squeeze()
    qt = tgt(next_states).amax(dim=1).squeeze()
    qm = rewards + (~dones * gamma * qt)
    print(f"DQN_LOSS = {T.mean((qm - qv) ** 2).item()!r}")

    # PPO - same weights, policy_old = current policy (ratio = 1)
    T.manual_seed(0)
    a5 = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2))
    T.manual_seed(1)
    c5 = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
    with T.no_grad():
        po = a5(states)
        vp = c5(states).squeeze()
        nvp = c5(next_states).squeeze()
        dp = rewards + (gamma * ~dones * nvp) - vp
        ap = discount(dp, dones, gamma * lambda_gae)
        rp = (ap + vp).detach().clone()
    logits5 = a5(states)
    d5 = dst.Categorical(logits=logits5)
    od = dst.Categorical(logits=po)
    lp5 = d5.log_prob(actions)
    lpo = od.log_prob(actions).detach().clone()
    ent5 = d5.entropy()
    v5 = c5(states).squeeze()
    ir = (lp5 - lpo).exp()
    tr = ir * ap
    cr_val = T.clamp(ir, 1 - eps_clip, 1 + eps_clip) * ap
    al5 = -T.min(tr, cr_val).mean()
    cl5 = beta_critic * T.mean((rp - v5) ** 2)
    el5 = -tau * T.mean(ent5)
    print(f"PPO_LOSS = {(al5 + cl5 + el5).item()!r}")


if __name__ == "__main__":
    main()
