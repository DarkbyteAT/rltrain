"""Tests for XptrackLogger — xptrack backend for rltrain tracking."""

from __future__ import annotations

from pathlib import Path

import pytest


xptrack = pytest.importorskip("xptrack", reason="xptrack not installed")

from xptrack.store.memory import InMemoryStore  # noqa: E402

from rltrain.tracking.backends.xptrack import XptrackLogger  # noqa: E402


@pytest.mark.unit
def test_lifecycle_start_log_finish() -> None:
    """Given an XptrackLogger, when start/log/finish, then metrics appear in the store."""
    # Given
    store = InMemoryStore()
    logger = XptrackLogger(store=store, project="test-project")

    # When
    logger.start(config={"lr": 0.001, "gamma": 0.99}, run_dir=Path("results/test/run-1"))
    logger.log_scalars({"return": 10.0, "length": 50.0}, step=1)
    logger.log_scalars({"return": 20.0, "length": 100.0}, step=2)
    logger.finish()

    # Then
    runs = store.query_runs(project="test-project")
    assert len(runs) == 1
    assert runs["name"][0] == "run-1"
    assert runs["status"][0] == "finished"

    metrics = store.query_metrics(runs["run_id"][0])
    assert len(metrics) == 4  # 2 metrics x 2 steps
    returns = metrics.filter(metrics["key"] == "return").sort("step")
    assert returns["value"].to_list() == [10.0, 20.0]
    assert returns["step"].to_list() == [1, 2]


@pytest.mark.unit
def test_log_hyperparams_writes_tags() -> None:
    """Given a started run, when log_hyperparams, then tags are persisted."""
    # Given
    store = InMemoryStore()
    logger = XptrackLogger(store=store, project="tag-test")
    logger.start(config={}, run_dir=Path("results/tag-test/run-hp"))

    # When
    logger.log_hyperparams({"lr": 0.001, "batch_size": 64, "algorithm": "PPO"})
    logger.finish()

    # Then
    runs = store.query_runs(project="tag-test")
    tags = runs["tags"][0]
    assert tags["lr"] == "0.001"
    assert tags["batch_size"] == "64"
    assert tags["algorithm"] == "PPO"


@pytest.mark.unit
def test_config_persisted_on_start() -> None:
    """Given config passed to start, then it appears in the run record."""
    # Given
    store = InMemoryStore()
    logger = XptrackLogger(store=store, project="config-test")

    # When
    logger.start(config={"gamma": 0.99}, run_dir=Path("results/config-test/run-cfg"))
    logger.finish()

    # Then
    runs = store.query_runs(project="config-test")
    config = runs["config"][0]
    assert config["gamma"] == 0.99


@pytest.mark.unit
def test_run_dir_stored_as_tag() -> None:
    """Given a run_dir, then it is stored as a tag on the run."""
    # Given
    store = InMemoryStore()
    logger = XptrackLogger(store=store, project="dir-test")

    # When
    logger.start(config={}, run_dir=Path("results/dir-test/run-42"))
    logger.finish()

    # Then
    runs = store.query_runs(project="dir-test")
    tags = runs["tags"][0]
    assert tags["run_dir"] == "results/dir-test/run-42"


@pytest.mark.unit
def test_finish_idempotent() -> None:
    """Given a finished run, calling finish again is a no-op."""
    # Given
    store = InMemoryStore()
    logger = XptrackLogger(store=store, project="idem-test")
    logger.start(config={}, run_dir=Path("results/idem-test/run-x"))

    # When
    logger.finish()
    logger.finish()  # should not raise

    # Then
    assert logger._run is None


@pytest.mark.unit
def test_log_before_start_is_noop() -> None:
    """Given no active run, log_scalars and log_hyperparams are silent no-ops."""
    # Given
    logger = XptrackLogger(store=":memory:", project="noop-test")

    # When / Then — should not raise
    logger.log_scalars({"x": 1.0}, step=0)
    logger.log_hyperparams({"y": 2})


@pytest.mark.unit
def test_tracking_callback_integration() -> None:
    """Given an XptrackLogger wired to TrackingCallback, the full pipeline works."""
    from rltrain.tracking.callback import TrackingCallback

    # Given
    store = InMemoryStore()
    logger = XptrackLogger(store=store, project="callback-test")
    callback = TrackingCallback(logger=logger, config={"algorithm": "PPO"})

    # When — simulate the callback lifecycle
    callback.on_train_start(
        agent=None,  # type: ignore[arg-type]
        env=None,  # type: ignore[arg-type]
        run_dir=Path("results/callback-test/run-cb"),
    )

    # Then — run was started, config and hyperparams logged
    runs = store.query_runs(project="callback-test")
    assert len(runs) == 1
    assert runs["status"][0] == "running"

    # Finish
    callback.on_train_end(
        agent=None,  # type: ignore[arg-type]
        env=None,  # type: ignore[arg-type]
        run_dir=Path("results/callback-test/run-cb"),
    )
    runs = store.query_runs(project="callback-test")
    assert runs["status"][0] == "finished"
