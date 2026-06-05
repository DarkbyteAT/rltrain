"""Agent implementations — pure-functional RL agents as Equinox modules."""

from rltrain.agents.advantage_ac import AdvantageAC as AdvantageAC
from rltrain.agents.agent import Agent as Agent
from rltrain.agents.agent import OnPolicyAgent as OnPolicyAgent
from rltrain.agents.agent import TrainState as TrainState
from rltrain.agents.agent import gradient_step as gradient_step
from rltrain.agents.distributional_dqn import DistributionalDQN as DistributionalDQN
from rltrain.agents.double_dqn import DoubleDQN as DoubleDQN
from rltrain.agents.ppo import PPO as PPO
from rltrain.agents.ppo_terminators import EpochTerminator as EpochTerminator
from rltrain.agents.ppo_terminators import KLEarlyStop as KLEarlyStop
from rltrain.agents.reinforce import REINFORCE as REINFORCE
from rltrain.agents.sac import SAC as SAC
from rltrain.agents.sac import SACState as SACState
from rltrain.agents.spo import SPO as SPO
from rltrain.agents.vanilla_ac import VanillaAC as VanillaAC
from rltrain.agents.vanilla_dqn import DQNState as DQNState
from rltrain.agents.vanilla_dqn import VanillaDQN as VanillaDQN
from rltrain.agents.vanilla_pg import VanillaPG as VanillaPG


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
