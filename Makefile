.PHONY: lint format format-check check typecheck test all

lint:
	uv run ruff check rltrain/

format:
	uv run ruff format rltrain/

format-check:
	uv run ruff format --check rltrain/

check:
	uv run ruff check --fix rltrain/

typecheck:
	uv run pyright rltrain/

test:
	uv run pytest tests/ -v

all: format-check lint typecheck test
