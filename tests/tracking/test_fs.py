"""Tests for FSLogger JSONL output."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def jsonl_path(tmp_path):
    return tmp_path / "metrics.jsonl"


@pytest.mark.unit
def test_writes_jsonl_records(jsonl_path):
    fsspec = pytest.importorskip("fsspec")  # noqa: F841
    from rltrain.tracking.backends.fs import FSLogger

    logger = FSLogger(str(jsonl_path))
    logger.start({}, Path("/tmp"))

    logger.log_scalars({"return": 15.3, "length": 120.0}, step=1)
    logger.log_scalars({"return": 20.1, "length": 130.0}, step=2)
    logger.finish()

    lines = jsonl_path.read_text().strip().split("\n")
    assert len(lines) == 2

    record_1 = json.loads(lines[0])
    assert record_1 == {"step": 1, "return": 15.3, "length": 120.0}

    record_2 = json.loads(lines[1])
    assert record_2 == {"step": 2, "return": 20.1, "length": 130.0}


@pytest.mark.unit
def test_finish_closes_file(jsonl_path):
    fsspec = pytest.importorskip("fsspec")  # noqa: F841
    from rltrain.tracking.backends.fs import FSLogger

    logger = FSLogger(str(jsonl_path))
    logger.start({}, Path("/tmp"))
    logger.finish()

    # After finish, internal state should be cleared
    assert logger._file is None


@pytest.mark.unit
def test_import_error_without_fsspec(monkeypatch):
    """FSLogger constructor raises ImportError when fsspec is missing."""
    import sys

    from rltrain.tracking.backends.fs import FSLogger

    # Remove fsspec from sys.modules so the import inside __init__ re-triggers
    monkeypatch.delitem(sys.modules, "fsspec", raising=False)

    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "fsspec":
            raise ImportError("No module named 'fsspec'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(ImportError, match="fsspec"):
        FSLogger("/tmp/test.jsonl")
