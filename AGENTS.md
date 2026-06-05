# AGENTS.md

Guidance for AI agents working with this codebase.

## Before Any Implementation

1. Read @CLAUDE.md for project context, architecture, and design decisions
2. Read @CONTRIBUTING.md for code conventions, architecture rules, and key patterns
3. Check existing implementations for patterns before writing new code

## Commands

```bash
# Environment
uv sync --group dev             # or: source .venv/bin/activate

# Quality
uv run ruff check rltrain/      # lint
uv run ruff format --check rltrain/   # format check
uv run pyright rltrain/         # type check (basic mode)
uv run pytest tests/            # default: unit + integration + e2e smoke
uv run pytest tests/ -m slow    # opt-in convergence suite
uv run pytest tests/ -m benchmark -s   # opt-in perf snapshot

# Before commit
make all                        # format-check + lint + typecheck + test
```

## Key Documentation

- @README.md — installation, usage, configuration, CLI, callbacks
- @CONTRIBUTING.md — code conventions, architecture rules, key patterns, PR workflow

## Critical Rules

### Agents

- The `Agent` Protocol is the only contract a trainer needs: `init(key) → state`, `learn(state, batch, key) → (state, metrics)`, `act(state, obs, key) → action`.
- Agents are `eqx.Module`s with **no array leaves** — pure architecture, hyperparameters as `eqx.field(static=True)`. JIT traces once.
- All mutable state lives in a `chex.dataclass` (`TrainState`, `DQNState`, `SACState`) threaded through `learn`. Never put params/opt_state on `self`.
- New optimisation techniques (sharpness-aware, regularisation hooks) belong in [samgria](https://github.com/DarkbyteAT/samgria), not in `learn()` or agent subclasses. Rltrain agents stay pure.
- All linear layers use orthogonal weight init via `rltrain.networks.MLP` (which wraps `eqx.nn.MLP` and re-initialises weights with `jax.nn.initializers.orthogonal()`).

### Configuration

- JSON + FQN — new agents/networks must be instantiable via `rltrain.builders.agent`.
- FQNs must resolve through `__init__.py` re-exports — use the shortest public name.
- Sub-objects with their own `fqn` are recursively built; PRNG sub-keys are auto-spliced into `key=` constructor parameters.
- No environment-specific dependencies — users plug gymnasium or gymnax envs from downstream scripts.

### Purity

- `learn` and `act` must compose with `jax.grad`, `jax.vmap`, and `jax.lax.scan`. If a method can't be scanned, it's in the wrong layer.
- Side effects (logging, file I/O, video recording) live in callbacks at segment boundaries — never inside `learn` or `act`.
- Use zero-filled sentinels (not `None`) for optional `Transition` fields — JAX needs fixed pytree structure.

### Testing

- Test structure mirrors the source layout (`tests/agents/` → `rltrain/agents/`, etc.)
- Given-When-Then structure
- Plain `def test_*` functions — no classes
- Every test marked with exactly one of `@pytest.mark.{unit,integration,e2e,benchmark,slow}` — see `tests/README.md`
- `pytest.ini` excludes `benchmark` and `slow` from the default run; convergence tests live in `test_foo_slow.py` files

### Code Style

- Python 3.11+ — `X | Y` union syntax, `list[T]`/`dict[K,V]` generics
- JAX idioms — `jax.random.key(seed)`, `jax.random.split(key)`, `eqx.partition/combine` for scannable modules
- `chex.dataclass` for state, `eqx.Module` for architecture
- `distreqx` for distributions; `dist.sample(key)` positional, **not** `seed=key` (that's distrax)
- Google-style docstrings

## Common Pitfalls

| Don't | Do |
|-------|-----|
| Put params or opt_state on `self` | Live in `TrainState` `chex.dataclass`, thread through `learn` |
| Mutate inside `learn` | Return a new state pytree |
| Hardcode optimisation pipeline | Use samgria's `GradientTransformation`-compatible primitives |
| Add env-specific dependencies | Keep rltrain generic — envs are plugged downstream |
| Use `None` for optional Transition fields | Zero-filled sentinels — pytree structure must be fixed |
| Call `dist.sample(seed=key)` | `dist.sample(key)` (distreqx, not distrax) |
| Use deep FQN paths | Re-export through `__init__.py`, use shortest public name |
| Wrap tests in classes | Plain `def test_*` functions |
| Put tests flat in `tests/` | Mirror the source layout |
| Add filetree diagrams to docs | Describe layout in prose |
