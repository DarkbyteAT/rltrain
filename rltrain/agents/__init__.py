"""Agent implementations — pure-functional RL agents as Equinox modules."""

from rltrain.agents.advantage_ac import AdvantageAC
from rltrain.agents.agent import Agent, OnPolicyAgent, TrainState, gradient_step
from rltrain.agents.distributional_dqn import DistributionalDQN
from rltrain.agents.double_dqn import DoubleDQN
from rltrain.agents.ppo import PPO
from rltrain.agents.ppo_terminators import EpochTerminator, KLEarlyStop
from rltrain.agents.reinforce import REINFORCE
from rltrain.agents.sac import SAC, SACState
from rltrain.agents.spo import SPO
from rltrain.agents.vanilla_ac import VanillaAC
from rltrain.agents.vanilla_dqn import DQNState, VanillaDQN
from rltrain.agents.vanilla_pg import VanillaPG


__all__ = [
    "AdvantageAC",
    "Agent",
    "DQNState",
    "DistributionalDQN",
    "DoubleDQN",
    "EpochTerminator",
    "KLEarlyStop",
    "OnPolicyAgent",
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
