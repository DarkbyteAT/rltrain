"""Unit tests for EpochTerminator protocol and KLEarlyStop."""

from typing import cast

import numpy as np
import pytest

from rltrain.agents.actor_critic import PPO, EpochTerminator, KLEarlyStop
from rltrain.env import MDP
from rltrain.env.trajectory import Trajectory
from tests.agents.conftest import EPS_CLIP, LAMBDA_GAE, make_ac_agent


# --- KLEarlyStop ---


@pytest.mark.unit
def test_kl_early_stop_triggers_above_threshold():
    # Given a terminator with target_kl=0.05
    terminator = KLEarlyStop(target_kl=0.05)

    # When approx_kl exceeds the threshold
    result = terminator.should_stop(0.06)

    # Then it should signal stop
    assert result is True


@pytest.mark.unit
def test_kl_early_stop_does_not_trigger_below_threshold():
    # Given a terminator with target_kl=0.05
    terminator = KLEarlyStop(target_kl=0.05)

    # When approx_kl is below the threshold
    result = terminator.should_stop(0.04)

    # Then it should not signal stop
    assert result is False


@pytest.mark.unit
def test_kl_early_stop_does_not_trigger_at_threshold():
    # Given a terminator with target_kl=0.05
    terminator = KLEarlyStop(target_kl=0.05)

    # When approx_kl equals the threshold exactly
    result = terminator.should_stop(0.05)

    # Then it should not stop (strict inequality)
    assert result is False


@pytest.mark.unit
def test_kl_early_stop_defaults_to_rollback():
    # Given a terminator constructed with default arguments
    terminator = KLEarlyStop(target_kl=0.05)

    # Then rollback should be True (KLE-Rollback is the default)
    assert terminator.rollback is True


@pytest.mark.unit
def test_kl_early_stop_kle_stop_mode():
    # Given a terminator with rollback disabled (KLE-Stop)
    terminator = KLEarlyStop(target_kl=0.05, rollback=False)

    # Then rollback should be False
    assert terminator.rollback is False


# --- Protocol conformance ---


@pytest.mark.unit
def test_kl_early_stop_satisfies_protocol():
    # Given a KLEarlyStop instance
    terminator = KLEarlyStop(target_kl=0.05)

    # Then it should satisfy the EpochTerminator protocol
    assert isinstance(terminator, EpochTerminator)


@pytest.mark.unit
def test_custom_terminator_satisfies_protocol():
    # Given a custom class with the right shape
    class AlwaysStop:
        rollback = False

        def should_stop(self, approx_kl: float) -> bool:
            return True

    # Then it should satisfy the protocol via structural subtyping
    assert isinstance(AlwaysStop(), EpochTerminator)


@pytest.mark.unit
def test_incomplete_class_does_not_satisfy_protocol():
    # Given a class missing the rollback attribute
    class MissingRollback:
        def should_stop(self, approx_kl: float) -> bool:
            return True

    # Then it should not satisfy the protocol
    assert not isinstance(MissingRollback(), EpochTerminator)


# --- PPO epoch loop integration ---
#
# These tests catch the lie of "KL is computed once per epoch, not once per
# mini-batch". They instrument PPO's step() loop with a spy on _approx_kl and
# assert the call count matches the claim. If a regression reintroduces the
# per-mini-batch call pattern, call_count jumps from num_epochs to
# num_epochs × num_minibatches and the test fails loudly.


class _StubMDP:
    """Minimal MDP stand-in that yields one trivial transition per ``step`` call."""

    @staticmethod
    def _trajectory() -> Trajectory:
        return Trajectory(
            state=np.zeros((1, 2), dtype=np.float32),
            action=np.zeros(1, dtype=np.int64),
            reward=np.zeros(1, dtype=np.float32),
            next_state=np.zeros((1, 2), dtype=np.float32),
            done=np.zeros(1, dtype=bool),
        )

    def step(self, _agent):
        return self._trajectory()


def _make_ppo(num_epochs: int, horizon: int, batch_size: int, **extra) -> PPO:
    return make_ac_agent(
        PPO,
        horizon=horizon,
        lambda_gae=LAMBDA_GAE,
        num_epochs=num_epochs,
        batch_size=batch_size,
        eps_clip=EPS_CLIP,
        **extra,
    )


def _prefill_memory(agent: PPO, n: int) -> None:
    for _ in range(n):
        agent.memory.append(_StubMDP._trajectory())


def _install_approx_kl_spy(agent: PPO) -> list[int]:
    """Replace ``agent._approx_kl`` with a counter-delegating wrapper.

    Returns a one-element list that will hold the running call count, so
    callers can read it after ``step()`` returns.
    """
    counter = [0]
    original = agent._approx_kl

    def spy(mini_batch):
        counter[0] += 1
        return original(mini_batch)

    agent._approx_kl = spy  # type: ignore[method-assign]
    return counter


@pytest.mark.unit
def test_approx_kl_called_once_per_epoch_not_once_per_minibatch():
    # Given a PPO agent with 2 epochs, 3 mini-batches per epoch, and a harmless terminator
    horizon, batch_size, num_epochs = 6, 2, 2
    agent = _make_ppo(
        num_epochs=num_epochs,
        horizon=horizon,
        batch_size=batch_size,
        # target_kl=1e9 guarantees the terminator never triggers, so all epochs run
        epoch_terminators=[KLEarlyStop(target_kl=1e9, rollback=False)],
    )
    _prefill_memory(agent, horizon - 1)
    counter = _install_approx_kl_spy(agent)

    # When step() triggers the full epoch loop (adds one trajectory, hits the horizon)
    agent.step(cast(MDP, _StubMDP()))

    # Then _approx_kl is called exactly once per epoch, not once per mini-batch.
    # Per-mini-batch would be num_epochs * 3 = 6; per-epoch is num_epochs = 2.
    assert counter[0] == num_epochs, (
        f"Expected {num_epochs} _approx_kl calls (once per epoch), got {counter[0]} "
        f"(per-mini-batch regression would give {num_epochs * 3})"
    )


@pytest.mark.unit
def test_approx_kl_not_called_when_no_epoch_terminators():
    # Given a PPO agent with no epoch terminators (the default, vanilla PPO)
    horizon, batch_size, num_epochs = 6, 2, 2
    agent = _make_ppo(
        num_epochs=num_epochs,
        horizon=horizon,
        batch_size=batch_size,
        epoch_terminators=(),
    )
    _prefill_memory(agent, horizon - 1)
    counter = _install_approx_kl_spy(agent)

    # When step() triggers the full epoch loop
    agent.step(cast(MDP, _StubMDP()))

    # Then _approx_kl is never called — there is nothing to check KL against,
    # so computing it would be pure waste. Vanilla PPO pays no KL overhead.
    assert counter[0] == 0, f"Expected 0 _approx_kl calls with no terminators, got {counter[0]}"
