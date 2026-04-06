# AGENTS.md

Guidance for AI agents working with this codebase.

## Before Any Implementation

1. Read @CLAUDE.md for project context, architecture, and design decisions
2. Read @CONTRIBUTING.md for code conventions, architecture rules, and key patterns
3. Check existing implementations for patterns before writing new code

## Commands

```bash
# Environment
source .venv/bin/activate       # or: uv sync --group dev

# Quality
uv run ruff check rltrain/     # lint
uv run pyright rltrain/        # type check (basic mode)
uv run pytest tests/           # test

# Before commit
make all                       # format-check + lint + typecheck + test
```

## Key Documentation

- @README.md — installation, usage, configuration, CLI, callbacks, experiment tracking
- @CONTRIBUTING.md — code conventions, architecture rules, key patterns, PR workflow
- @tests/README.md — test organisation, conventions, running tests

### Module READMEs

Before working in a module, read its README for architecture, protocols, and how-to-add guides.

| Module | README | Covers |
|--------|--------|--------|
| `rltrain/agents/` | @rltrain/agents/README.md | Agent hierarchy, template method pattern, how to add new algorithms |
| `rltrain/callbacks/` | @rltrain/callbacks/README.md | Callback protocol, built-in callbacks, how to write custom ones |
| samgria (external) | [samgria docs](https://github.com/DarkbyteAT/samgria) | GradientTransform protocol, SAM/ASAM/LAMP, how to add new transforms |
| `rltrain/tracking/` | @rltrain/tracking/README.md | TrackingCallback, MetricsLogger protocol, backend implementations |
| `rltrain/utils/` | @rltrain/utils/README.md | FQN builder system, device resolution, math helpers |
| `rltrain/env/` | @rltrain/env/README.md | MDP wrapper, Trajectory dataclass |

## Critical Rules

### Agents

- Agent inheritance chain is deliberate — each level adds one concept
- `Agent.learn()` is the single orchestration point for optimisation
- New optimisation techniques go in [samgria](https://github.com/DarkbyteAT/samgria), not in `learn()` or subclasses
- All networks must use orthogonal weight initialisation

### Configuration

- JSON + FQN — new agents/networks must be instantiable via the FQN builder
- FQNs must resolve through `__init__.py` re-exports — use the shortest public name
- No environment-specific dependencies — users plug gymnasium envs from downstream scripts

### Testing

- Test structure mirrors the source layout (`tests/agents/` → `rltrain/agents/`, etc.)
- Given-When-Then structure
- Plain `def test_*` functions — no classes

### Code Style

- Python 3.11+ — `X | Y` union syntax, `list[T]`/`dict[K,V]` generics
- PyTorch aliases: `T` for `torch`, `dst` for `torch.distributions`, `F` for `torch.nn.functional`
- NumPy-style docstrings

## Common Pitfalls

| Don't | Do |
|-------|-----|
| Hardcode optimisation in `learn()` | Use `GradientTransform` from [samgria](https://github.com/DarkbyteAT/samgria) |
| Add env-specific dependencies | Keep rltrain generic — envs are plugged downstream |
| Use deep FQN paths | Re-export through `__init__.py`, use shortest public name |
| Wrap tests in classes | Use plain `def test_*` functions |
| Put tests flat in `tests/` | Mirror the source layout |
