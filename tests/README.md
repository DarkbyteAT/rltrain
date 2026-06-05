# Tests

## Philosophy

Test behaviour, not implementation. Tests prove the code does what it should, catch regressions, and serve as executable examples of correct usage. Follow the Given-When-Then pattern for readability.

## Organisation

The test structure mirrors the source layout. Each subdirectory under `tests/` corresponds to a module in the source tree. When adding tests for a new module, create a matching subdirectory under `tests/`.

Shared test helpers and fixtures live in `tests/conftest.py` (global) and per-directory `conftest.py` files for scoped fixtures.

Per-file convention: `test_foo.py` for the fast variant of a module's coverage, `test_foo_slow.py` for the heavier variant (real training to convergence, large tensors, etc). The `_slow` suffix is the indicator; the marker enforces the exclusion.

## Test Markers

Every test must be marked with exactly one of the five markers. `pytest.ini` excludes `benchmark` and `slow` from the default run via `addopts = -m "not benchmark and not slow"`, so the default `pytest tests/` invocation runs only the fast wiring layer.

| Marker | When to use | Default? |
|---|---|---|
| `@pytest.mark.unit` | A single function or class in isolation, mocked dependencies | on |
| `@pytest.mark.integration` | Tests crossing module boundaries with real dependencies | on |
| `@pytest.mark.e2e` | Full pipeline smoke tests — wire the trainer end-to-end on a small num_steps and assert the pipeline runs without producing non-finite state | on |
| `@pytest.mark.benchmark` | Forward-pass timing / compile-time / wall-clock probes; print data for PR review | off (skipped by default) |
| `@pytest.mark.slow` | Takes >5s on CPU — convergence tests, exhaustive parameter sweeps | off (skipped by default) |

Convergence claims belong in the `slow` file; the smoke file in the same module asserts only that the pipeline composes. See `tests/test_e2e_training.py` (smoke) vs `tests/test_e2e_training_slow.py` (convergence) for the canonical pairing.

## Running Tests

```bash
source .venv-spike/bin/activate          # or: uv sync --group dev

pytest tests/                            # default: unit + integration + e2e smoke
pytest tests/ -m unit                    # unit only — the pre-commit fast path
pytest tests/ -m slow                    # convergence suite (~7 min)
pytest tests/ -m benchmark -s            # perf snapshot, with printed numbers
pytest tests/ -m "not benchmark"         # everything except the perf snapshot
```

`make test` runs the default set (inherits `addopts`); `make test-slow` and `make test-benchmark` run the opt-in suites.

Pre-commit hooks invoke `pytest -m unit` only — fast feedback for routine commits. CI runs the default set; the slow and benchmark suites are invoked on demand or on a separate schedule.

## Conventions

- Plain `def test_*` functions — no test classes.
- Given-When-Then structure for all tests.
- Helper modules prefixed with `_` to distinguish from test files (e.g. `tests/agents/_helpers.py`).
- Mark every test — unmarked tests don't run in pre-commit and aren't surfaced in default reports.
- All tests must pass before committing.
