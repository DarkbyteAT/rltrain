"""Lightweight fixtures for testing callbacks without a full training loop."""

from dataclasses import dataclass, field

import pytest
import torch.nn as nn


@dataclass
class StubAgent:
    """Minimal agent-like object with a real nn.ModuleDict for state_dict."""

    name: str = "TestAgent"
    model: nn.ModuleDict = field(default_factory=lambda: nn.ModuleDict({
        "actor": nn.Sequential(nn.Linear(4, 2)),
    }))


@dataclass
class StubEnv:
    """Minimal env-like object with the metric attributes callbacks read."""

    episode_count: int = 0
    episode_steps: int = 0
    total_steps: int = 0
    run_beta: float = 0.1
    target_reward: float | None = None
    length_history: list[int] = field(default_factory=list)
    return_history: list[float] = field(default_factory=list)
    run_history: list[float] = field(default_factory=list)


@pytest.fixture
def stub_agent():
    return StubAgent()


@pytest.fixture
def stub_env():
    return StubEnv()


@pytest.fixture
def populated_env():
    """An env that looks like it completed 3 episodes."""
    return StubEnv(
        episode_count=3,
        episode_steps=150,
        total_steps=150,
        length_history=[40, 55, 55],
        return_history=[20.0, 35.0, 42.0],
        run_history=[20.0, 28.5, 36.15],
    )
