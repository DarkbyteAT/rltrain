"""Unit tests for EpochTerminator protocol and KLEarlyStop."""

import pytest

from rltrain.agents.actor_critic import EpochTerminator, KLEarlyStop


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
