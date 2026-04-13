"""Agent implementations — pure-functional RL agents as Equinox modules."""

from spike.agents.advantage_ac import AdvantageAC
from spike.agents.agent import Agent, OnPolicyAgent, TrainState, gradient_step
from spike.agents.distributional_dqn import DistributionalDQN
from spike.agents.double_dqn import DoubleDQN
from spike.agents.ppo import PPO
from spike.agents.reinforce import REINFORCE
from spike.agents.sac import SAC, SACState
from spike.agents.spo import SPO
from spike.agents.vanilla_ac import VanillaAC
from spike.agents.vanilla_dqn import DQNState, VanillaDQN
from spike.agents.vanilla_pg import VanillaPG


__all__ = [
    "AdvantageAC",
    "Agent",
    "DQNState",
    "OnPolicyAgent",
    "DistributionalDQN",
    "DoubleDQN",
    "PPO",
    "REINFORCE",
    "SAC",
    "SACState",
    "SPO",
    "TrainState",
    "VanillaAC",
    "VanillaDQN",
    "VanillaPG",
    "gradient_step",
]
