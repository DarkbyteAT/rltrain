# CLAUDE.md

Read [README.md](README.md) for the full framework overview — architecture, algorithms, configuration system, and usage.

## Project Context

RLTrain is a PyTorch deep RL framework originally built for a 2022 dissertation (COMP3200, University of Southampton). It is being revived as an active research tool, open-source framework, and educational resource.

## Architecture at a Glance

- **Agent hierarchy**: `Agent` (ABC) → `VanillaPG` → `REINFORCE` → `VanillaAC` → `AdvantageAC` → `PPO`, plus `VanillaDQN`. Each subclass adds exactly one concept. Template method pattern: `learn()` orchestrates, subclasses override `loss()` and `descend()`.
- **Environment**: `MDP` wraps `gymnasium.vector.SyncVectorEnv` with auto-reset, metric tracking, and optional channel preprocessing.
- **Configuration**: JSON files with `fqn` fields for dynamic class resolution. Agents, networks, optimisers, and wrappers are all specified declaratively.
- **Neural networks**: `mlp`, `cnn`, `SkipMLP` (D2RL), `RFF` — all use orthogonal init.
- **Robust optimisation**: SAM and LAMP baked into `Agent.learn()`, toggled via config.
- **Trainer + Callbacks**: `Trainer` owns the training loop and orchestrates `Callback` hooks (on_train_start, on_step, on_episode_end, on_checkpoint, on_train_end). Built-in callbacks handle CSV logging, SVG plots, and model checkpoints. Custom callbacks implement the `Callback` protocol.
- **Entry point**: `run.py` is a thin CLI wrapper — parses args, builds objects, delegates to `Trainer.fit()`.

## Directory Structure

```
rltrain/              # Framework package
├── agents/           # Agent implementations (policy_gradient/, actor_critic/, q_learning/)
├── callbacks/        # Callback protocol + built-in callbacks (checkpoint, csv_logger, plot)
├── env/              # MDP wrapper + Trajectory dataclass
├── nn/               # Network modules (mlp, cnn, d2rl, rff)
├── trainer.py        # Trainer class — training loop + callback orchestration
└── utils/            # Builders (FQN loader), discount, center, grad, lerp

{cartpole,acrobot,breakout,invaders,pixelcopter,catcher}/
                      # Experiment configs (env.json + agent variants)
run.py                # Thin CLI wrapper — arg parsing, object creation, calls Trainer.fit()
```

## Code Conventions

- **Python 3.10+** — uses `X | Y` union syntax, `list[T]`/`dict[K,V]` generics
- **PyTorch** — `T` alias for `torch`, `dst` for `torch.distributions`, `F` for `torch.nn.functional`
- **NumPy-style docstrings** with backtick-wrapped parameter names
- **Orthogonal weight init** on all linear and conv layers
- **No tests exist yet** — when adding tests, use plain `async def test_*` functions with pytest, no classes

## Key Patterns

### FQN Builder System
The `load(fqn)` function in `utils/builders/` dynamically imports any class by fully-qualified name. This is the core of the configuration system — JSON configs specify `"fqn": "rltrain.agents.actor_critic.PPO"` and the builder resolves it at runtime.

### Agent Template Method
`Agent.learn()` handles the full optimisation step including optional SAM/LAMP perturbation. Subclasses only need to implement:
- `setup()` — initialise networks and optimisers
- `act(states)` → Distribution
- `step(env)` — collect experience, decide when to learn
- `load()` → batch tensors from memory
- `loss(*batch)` → scalar loss
- `descend()` — optimizer step + gradient clipping

### Callback Protocol
`rltrain.callbacks.Callback` is a `@runtime_checkable` Protocol with five hook methods, all defaulting to no-op (`...`). Built-in callbacks: `CheckpointCallback`, `CSVLoggerCallback`, `PlotCallback`. The `Trainer` accepts a `callbacks` list — if None, it uses the three built-ins. Custom callbacks implement any subset of the protocol methods.

### Config-Driven Composition
Networks are built by composing modules listed in JSON. Each model entry becomes a layer in `nn.Sequential`. Optimisers are stored as factory lambdas until `setup()` binds them to parameters.

## Known Technical Debt

- Uses **gymnasium** — 5-tuple `step()` returns with `terminated | truncated` combined into `done` at the MDP level
- **Packaging** — `pyproject.toml` exists with hatchling backend; install via `pip install -e ".[dev]"` or `uv sync --group dev`
- **Single-env vectorisation** — wraps 1 env in `SyncVectorEnv` (no true parallelism)
- **No experiment tracking** integration (WandB, TensorBoard)
- **No CI/CD** pipeline
- Experiment configs mixed with framework code at repo root

## Working With This Codebase

- The JSON config system is the heart of the framework. When adding new agents or networks, ensure they can be instantiated via the FQN builder with keyword arguments from JSON.
- The agent inheritance chain is deliberate — each level adds one concept. Maintain this when adding algorithms.
- `Agent.learn()` should remain the single orchestration point for optimisation. New optimisation techniques (like SAM/LAMP) go here, not in subclasses.
- All networks should use orthogonal weight initialisation.
- Respect the separation: framework code in `rltrain/`, experiment configs in named directories, results in `dump/`.

## References

See [references/references.bib](references/references.bib) for the academic papers behind each algorithm and technique.
