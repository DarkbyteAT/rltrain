# Contributing to RLTrain

## Development Setup

```bash
git clone https://github.com/DarkbyteAT/rltrain.git
cd rltrain
uv sync --group dev             # creates .venv, installs deps
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Code Conventions

- **Python 3.11+** — `X | Y` union syntax, `list[T]`/`dict[K,V]` generics
- **JAX idioms** — `jax.random.key(seed)`, `jax.random.split(key)`, `eqx.partition/combine` for scannable modules, `chex.dataclass` for state pytrees
- **Distributions** — `distreqx`; `dist.sample(key)` positional (not `seed=key`)
- **Google-style docstrings** with LaTeX math (`$...$`, `$$...$$`). Prefix raw strings (`r"""..."""`) when docstrings contain LaTeX backslashes.
- **Orthogonal weight init** on linear layers via `rltrain.networks.MLP`
- **Type-check surface uses `chex.Array`** (not `jax.Array`) so callers can pass numpy arrays at the boundary without narrowing complaints. Internals can use `jax.Array` once values are lifted via `jnp.asarray`.

## Testing & Documentation

Tests and docs serve overlapping but distinct purposes — both must exist:

- **Tests verify implementation** — prove the code does what it should, catch regressions, double as examples.
- **Documentation declares intent** — what the code *should* do, why, and how to use it.
- **The duplication is the point.** Inconsistency between them forces the question: did intent change, or did implementation drift?

Both must be updated alongside code changes in the same PR.

### Tests

- Test structure mirrors the source layout (`tests/agents/` → `rltrain/agents/`)
- Plain `def test_*` functions — no classes
- Given-When-Then structure
- Every test marked with exactly one of `@pytest.mark.{unit,integration,e2e,benchmark,slow}` — see `tests/README.md` for marker semantics
- `pytest.ini` excludes `benchmark` and `slow` from the default run; convergence tests live in `test_foo_slow.py` files

```bash
uv run pytest tests/                    # default: unit + integration + e2e smoke
uv run pytest tests/ -m slow            # opt-in convergence suite
uv run pytest tests/ -m benchmark -s    # opt-in perf snapshot
make test-slow                          # same as -m slow
make test-benchmark                     # same as -m benchmark -s
```

## Linting & Type Checking

```bash
uv run ruff check rltrain/             # lint
uv run ruff format --check rltrain/    # format check
uv run pyright rltrain/                # type check (basic mode)
```

Tool configs live in dedicated files (`ruff.toml`, `pytest.ini`, `pyrightconfig.json`), not in `pyproject.toml`.

A Makefile wraps these: `make lint`, `make format`, `make typecheck`, `make test` (default test set), `make test-slow` (convergence suite), `make test-benchmark` (perf snapshot), or `make all` for the full gate (format-check → lint → typecheck → test). `make fix` auto-fixes lint violations.

Pyright runs in `basic` mode against `rltrain/` only. Most categories are downgraded to `warning` so CI passes on warnings — keep new code error-free, and reduce warnings when you touch a file.

## Pull Request Workflow

1. Create a PR with a clear description (see template below).
2. Run `/gemini review` for an automated pass.
3. Resolve or respond to **all** Gemini comments — no unaddressed feedback.
4. Re-run `/gemini review` until convergence.
5. Then request human review.
6. Squash-merge with `--delete-branch` once approved.

PRs must not merge with unresolved automated review comments.

## Directory Structure

The package lives under `rltrain/` with submodules for `agents/`, `callbacks/`, `builders/`, `trainer/`, plus flat modules (`buffer.py`, `transitions.py`, `heads.py`, `distributions.py`, `networks.py`, `env.py`, `math.py`, `cli.py`). Tests under `tests/` mirror the source layout. Example configs live in `examples/`. Run `tree -L 2 rltrain/` for the current layout.

## Architecture Rules

- **Framework code in `rltrain/`**, experiment configs in `examples/`, results in `<dump>/`.
- **`Agent` Protocol** is the contract — `init/learn/act`. Three methods, agnostic to algorithm family.
- **No mutable state on agent modules.** Agents are `eqx.Module`s with static fields only. All mutable state in `TrainState`/`DQNState`/`SACState` pytrees.
- **`learn` and `act` must be jittable** — composable with `grad`, `vmap`, `lax.scan`. If a method needs `pure_callback` or `io_callback`, it's in the wrong layer.
- **New optimisation techniques** (sharpness-aware, parameter regularisation) belong in [samgria](https://github.com/DarkbyteAT/samgria) as JAX-native gradient transforms. Rltrain agents stay pure.
- **All linear layers** must use orthogonal weight init.
- **JSON + FQN** — new agents and networks must be instantiable via `rltrain.builders.agent` with kwargs from JSON.
- **No environment-specific dependencies** — rltrain is general-purpose. Plug envs from downstream scripts.

## Key Patterns

### FQN Builder System

`rltrain.builders.agent.agent(cfg, key)` dynamically resolves any `fqn` field to a Python class and recursively constructs sub-objects (`actor`, `action_head`, `critic`, `optimizer`, `epoch_terminators`). PRNG sub-keys are auto-spliced into constructors that accept a `key=` parameter (detected via `inspect`). FQNs must resolve through `__init__.py` re-exports — use the shortest public name (e.g. `rltrain.agents.PPO`, not `rltrain.agents.ppo.PPO`).

### Agent Protocol

`rltrain.agents.Agent` is a `runtime_checkable` `Protocol`:

- `init(key) → state` — fresh `TrainState` pytree
- `learn(state, batch, key) → (state, metrics)` — pure functional update; returns new state and a dict of scalar metrics
- `act(state, obs, key) → action` — policy

Subclasses of `OnPolicyAgent` only need to override `_loss(...)`. DQN variants override `_loss` against `dqn_learn_step`. SAC and DistributionalDQN are standalone modules implementing the Protocol directly.

### Callback Protocol

`rltrain.callbacks.Callback` is a `@runtime_checkable` `Protocol` with five hooks (all default no-op):

| Hook | Signature | Called |
|---|---|---|
| `on_train_start` | `(config, run_dir)` | once, before the loop |
| `on_step` | `(step, metrics: dict[str, float])` | every step |
| `on_episode_end` | `(episode, episode_return, episode_length, running_return)` | episode boundaries |
| `on_checkpoint` | `(step, agent_state, run_dir)` | every `checkpoint_steps` |
| `on_train_end` | `(agent_state, run_dir)` | once, after the loop |

Hooks fire **Python-side at segment boundaries** — `ScanLoop` collects `StepOutput` arrays inside `lax.scan` and dispatches at each `checkpoint_steps` segment. This is a deliberate departure from the original `io_callback` design (see `rltrain/trainer/_loops.py`).

### Buffer

`ExperienceBuffer` is a single `chex.dataclass` covering rollout (drain at horizon), horizon mini-batch (PPO/SPO), and persistent replay (DQN/SAC) regimes via configuration. PER is a buffer option (`prioritised=True`), not a separate type. Priorities are always allocated; the uniform path returns `is_weights=jnp.ones(...)`. All ops are pure JAX and jittable.

### Loop Strategies

The `TrainingLoop` `Protocol` has three implementations:

- `PythonLoop` — Python `while` loop; works for both gymnasium and gymnax envs
- `ScanLoop` — `lax.scan` segments; auto-selected when `env.capabilities.scan_rollout` is True
- `PmapLoop` — multi-device via `jax.pmap`; falls back to ScanLoop on single device

`Trainer` selects the right loop from `env.capabilities` unless one is supplied explicitly.

## References

See [references/references.bib](references/references.bib).
