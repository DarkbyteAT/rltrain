"""Tests for TrackingCallback hook mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from rltrain.tracking.callback import TrackingCallback


@dataclass
class SpyLogger:
    """Records every call made to it for assertion."""

    calls: list[tuple[str, tuple]] = field(default_factory=list)

    def start(self, config: dict[str, Any], run_dir: Path) -> None:
        self.calls.append(("start", (config, run_dir)))

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        self.calls.append(("log_scalars", (metrics, step)))

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        self.calls.append(("log_hyperparams", (params,)))

    def finish(self) -> None:
        self.calls.append(("finish", ()))


@dataclass
class StubAgent:
    name: str = "TestAgent"


@dataclass
class StubEnv:
    episode_count: int = 0
    length_history: list[int] = field(default_factory=list)
    return_history: list[float] = field(default_factory=list)
    run_history: list[float] = field(default_factory=list)


@pytest.fixture
def spy_logger():
    return SpyLogger()


@pytest.fixture
def config():
    return {"fqn": "rltrain.agents.actor_critic.PPO", "gamma": 0.99}


@pytest.fixture
def stub_agent():
    return StubAgent()


@pytest.fixture
def stub_env():
    return StubEnv(
        episode_count=1,
        length_history=[100],
        return_history=[42.0],
        run_history=[38.5],
    )


@pytest.mark.unit
def test_on_train_start_calls_start_and_log_hyperparams(spy_logger, config, stub_agent, stub_env):
    cb = TrackingCallback(spy_logger, config)
    run_dir = Path("/tmp/run_1")

    cb.on_train_start(stub_agent, stub_env, run_dir)

    assert spy_logger.calls[0] == ("start", (config, run_dir))
    assert spy_logger.calls[1] == ("log_hyperparams", (config,))
    assert len(spy_logger.calls) == 2


@pytest.mark.unit
def test_on_episode_end_logs_scalars(spy_logger, config, stub_agent, stub_env):
    cb = TrackingCallback(spy_logger, config)

    cb.on_episode_end(stub_agent, stub_env, episode=5)

    assert len(spy_logger.calls) == 1
    name, (metrics, step) = spy_logger.calls[0]
    assert name == "log_scalars"
    assert step == 5
    assert metrics == {"return": 42.0, "length": 100, "running_return": 38.5}


@pytest.mark.unit
def test_on_train_end_calls_finish(spy_logger, config, stub_agent, stub_env):
    cb = TrackingCallback(spy_logger, config)

    cb.on_train_end(stub_agent, stub_env, Path("/tmp/run_1"))

    assert spy_logger.calls == [("finish", ())]


@pytest.mark.unit
def test_on_step_is_noop(spy_logger, config, stub_agent, stub_env):
    cb = TrackingCallback(spy_logger, config)

    cb.on_step(stub_agent, stub_env, step=10)

    assert spy_logger.calls == []


@pytest.mark.unit
def test_on_checkpoint_is_noop(spy_logger, config, stub_agent, stub_env):
    cb = TrackingCallback(spy_logger, config)

    cb.on_checkpoint(stub_agent, stub_env, Path("/tmp/run_1"))

    assert spy_logger.calls == []
