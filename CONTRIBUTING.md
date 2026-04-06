# Contributing to RLTrain

## Development Setup

```bash
git clone https://github.com/DarkbyteAT/rltrain.git
cd rltrain
source scripts/enable-venv.sh   # creates venv, installs deps (uses uv if available, pip otherwise)
```

Or manually:

```bash
# With uv (recommended) — creates .venv automatically
uv sync --group dev

# With pip
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Code Conventions

- **Python 3.10+** — `X | Y` union syntax, `list[T]`/`dict[K,V]` generics
- **PyTorch aliases** — `T` for `torch`, `dst` for `torch.distributions`, `F` for `torch.nn.functional`
- **NumPy-style docstrings** with backtick-wrapped parameter names
- **Orthogonal weight init** on all linear and conv layers

## Testing & Documentation

Tests and documentation serve overlapping but distinct purposes — both must exist, and the overlap is intentional:

- **Tests verify implementation** — they prove the code does what it should, catch regressions, and serve as executable examples of correct usage.
- **Documentation declares intent** — it describes what the code *should* do, why it exists, and how to use it.
- **The duplication is the point.** When tests and docs describe the same behaviour, any inconsistency between them forces you to ask: did the intent change, or did the implementation drift? That tension is a feature, not waste.

Both must be updated alongside code changes in the same PR.

### Tests

See [tests/README.md](tests/README.md) for the full testing guide. Key points:

- Test structure mirrors the source layout (`tests/agents/` → `rltrain/agents/`, etc.)
- Plain `def test_*` functions — no classes
- Given-When-Then structure

```bash
uv run pytest tests/ -v
```

## Linting & Type Checking

```bash
uv run ruff check rltrain/          # lint
uv run ruff format --check rltrain/ # format check
uv run pyright rltrain/             # type check (basic mode)
```

Tool configs live in separate files (`ruff.toml`, `pytest.ini`, `pyrightconfig.json`), not in `pyproject.toml`.

A Makefile wraps these commands for convenience: `make lint`, `make format`, `make typecheck`, `make test`, or `make all` to run the full quality gate (format-check → lint → typecheck → test). Run `make fix` to auto-fix lint violations.

## Pull Request Workflow

1. Create a PR with a clear description (see PR template below).
2. Run `/gemini review` to request an automated review.
3. Resolve or respond to **all** Gemini comments — do not leave unaddressed feedback.
4. Re-run `/gemini review` after changes until the review converges (no new substantive comments).
5. Only then request human review.
6. Squash-merge with `--delete-branch` once approved.

PRs must not be merged with unresolved automated review comments. The Gemini review cycle is a quality gate, not a suggestion.

## Directory Structure

- **`rltrain/`** — framework package: agents (policy_gradient/, actor_critic/, q_learning/), callbacks, env (MDP wrapper), nn (network modules), transforms (re-exported from samgria: SAM/ASAM/LAMP), utils (builders, device, math helpers), and `trainer.py`
- **`examples/`** — experiment configs (env.json + agent variants per environment)
- **`tests/`** — test suite mirroring the source layout (see [tests/README.md](tests/README.md))
- **`run.py`** — thin CLI wrapper

## Architecture Rules

- **Framework code in `rltrain/`**, experiment configs in `examples/`, results in `dump/`.
- **Agent inheritance chain** is deliberate — each level adds one concept. Maintain this when adding algorithms.
- **`Agent.learn()`** is the single orchestration point for optimisation. New optimisation techniques are implemented as `GradientTransform` classes in `rltrain/transforms/`, not hardcoded in `learn()` or subclasses.
- **All networks** must use orthogonal weight initialisation.
- **JSON + FQN** — new agents/networks must be instantiable via the FQN builder with keyword arguments from JSON.
- **No environment-specific dependencies** — rltrain is a general-purpose RL framework. Users plug gymnasium-compatible environments from downstream scripts.

## Key Patterns

### FQN Builder System

The `load(fqn)` function in `utils/builders/` dynamically imports any class by fully-qualified name. JSON configs specify `"fqn": "rltrain.agents.actor_critic.PPO"` and the builder resolves it at runtime. FQNs must resolve through `__init__.py` re-exports — use the shortest public name (e.g. `rltrain.nn.SkipMLP`, not `rltrain.nn.d2rl.SkipMLP`).

### Agent Template Method

`Agent.learn()` handles the full optimisation step. Subclasses only need to implement:

- `setup()` — initialise networks and optimisers
- `act(states)` → Distribution
- `step(env)` — collect experience, decide when to learn
- `load()` → batch tensors from memory
- `loss(*batch)` → scalar loss
- `descend()` — optimizer step + gradient clipping

### Gradient Transform Pipeline

`Agent.learn()` applies a composable pipeline of `GradientTransform` steps between `loss.backward()` and `descend()`. Each transform implements a two-phase protocol:

- `apply(model, loss_fn, batch)` — **pre-descent** hook for transforms that modify gradients or temporarily perturb parameters (e.g. SAM, ASAM).
- `post_step(model)` — **post-descent** hook for transforms that operate on updated parameters (e.g. LAMPRollback noise injection + rollback).

The pipeline is configured via the `grad_transforms` key in agent JSON, using the FQN resolver:

```json
"grad_transforms": [
    {"fqn": "samgria.SAM", "rho": 1e-2},
    {"fqn": "samgria.LAMPRollback", "eps": 5e-3, "rollback_len": 10}
]
```

Omitting `grad_transforms` (or passing an empty list) gives vanilla gradient descent.

Built-in transforms are provided by [samgria](https://github.com/DarkbyteAT/samgria) and re-exported via `rltrain/transforms/`:

| Transform | Phase | Description |
|-----------|-------|-------------|
| `SAM` | pre-descent | Sharpness-Aware Minimisation — perturb in gradient direction, recompute loss at worst-case point |
| `ASAM` | pre-descent | Adaptive SAM — perturbation scaled by parameter magnitude for scale invariance |
| `LAMPRollback` | post-descent | Noise injection + moving average rollback for flat-minima exploration |

To add a custom transform, implement a class with `apply()` and `post_step()` methods matching the `GradientTransform` protocol, place it anywhere importable, and reference it via FQN in the config.

### Callback Protocol

`rltrain.callbacks.Callback` is a `@runtime_checkable` Protocol with five hook methods, all defaulting to no-op (`...`). The `Trainer` accepts a `callbacks` list — if None, it uses the three built-ins (`CSVLoggerCallback`, `PlotCallback`, `CheckpointCallback`). Custom callbacks implement any subset of the protocol methods.

### Device System

rltrain is device-agnostic. The `Agent` constructor takes a `T.device` and all tensor operations use `.to(self.device)`. The CLI exposes `--device {cpu,cuda,mps,auto}` (default: `auto`). Resolution logic lives in `rltrain.utils.device.resolve_device()`, which auto-detects: CUDA → MPS → CPU. Programmatic users call `resolve_device("auto")` or pass a `T.device` directly to the builder.

## References

See [references/references.bib](references/references.bib) for the academic papers behind each algorithm and technique.
