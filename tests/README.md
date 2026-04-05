# Tests

## Philosophy

Test behaviour, not implementation. Tests prove the code does what it should, catch regressions, and serve as executable examples of correct usage. Follow the Given-When-Then pattern for readability.

## Organisation

The test structure mirrors the source layout. Each subdirectory under `tests/` corresponds to a module in the source tree — `tests/agents/` tests `rltrain/agents/`, `tests/callbacks/` tests `rltrain/callbacks/`, and so on. When adding tests for a new module, create a matching subdirectory under `tests/`.

Shared test helpers and fixtures live in `tests/conftest.py` (global) and per-directory `conftest.py` files for scoped fixtures (e.g. `tests/agents/conftest.py` provides agent builder fixtures).

## Conventions

- Plain `def test_*` functions — no test classes
- Given-When-Then structure for all tests
- Agent tests use the fixtures in `tests/agents/conftest.py` for consistent setup
- Helper modules (e.g. `tests/agents/_compute_expected.py`) are prefixed with `_` to distinguish from test files

## Running Tests

```bash
source .venv/bin/activate
pytest tests/                     # all tests
pytest tests/agents/              # agent tests only
pytest tests/callbacks/           # callback tests only
pytest tests/tracking/            # tracking tests only
pytest tests/ -q                  # quiet output
```

All tests must pass before committing. Pre-commit hooks enforce this.
