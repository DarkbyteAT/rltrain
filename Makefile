.PHONY: lint format format-check fix typecheck test test-slow test-benchmark all

lint:
	uv run ruff check rltrain/

format:
	uv run ruff format rltrain/

format-check:
	uv run ruff format --check rltrain/

fix:
	uv run ruff check --fix rltrain/

typecheck:
	uv run pyright rltrain/

# Default test set (unit + integration + e2e smoke).
# Inherits the -m "not benchmark and not slow" exclusion from pytest.ini.
test:
	uv run pytest tests/ -v

# Opt-in convergence suite — explicit marker overrides the default exclusion.
test-slow:
	uv run pytest tests/ -m slow -v

# Opt-in perf snapshot — explicit marker; -s passes through the printed numbers.
test-benchmark:
	uv run pytest tests/ -m benchmark -s -v

all: format-check lint typecheck test
