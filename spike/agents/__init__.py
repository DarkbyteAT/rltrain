"""Agent implementations — pure-functional RL agents as Equinox modules."""

from spike.agents.agent import Agent, TrainState, gradient_step
from spike.agents.vanilla_dqn import DQNState, VanillaDQN
from spike.agents.vanilla_pg import VanillaPG


__all__ = [
    "Agent",
    "DQNState",
    "TrainState",
    "VanillaDQN",
    "VanillaPG",
    "gradient_step",
]
