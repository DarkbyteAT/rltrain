# Design Spec: Loss Tests, CI/CD, Experiment Tracking

Three independent cards designed together because they share no file conflicts and can be implemented in parallel.

---

## Card 19 — Agent Loss Function + SAM/LAMP Tests

### Problem

RL loss functions fail silently. A sign flip, a missing `.detach()`, or a wrong discount factor still produces a scalar loss — backprop runs, the agent "trains," but learns nonsense. Hand-computed expected values are the only reliable way to catch these bugs.

### Structure

```
tests/agents/
├── __init__.py
├── conftest.py          # Minimal agent factory, seeded tensor builders
├── test_vanilla_pg.py
├── test_reinforce.py
├── test_vanilla_ac.py
├── test_a2c.py
├── test_ppo.py
├── test_dqn.py
└── test_sam_lamp.py
```

### Test Strategy

Each agent test:

1. **Seed everything.** `torch.manual_seed(0)`, deterministic mode.
2. **Build a tiny agent.** 2-input, 2-output, single hidden layer of 4 units. Orthogonal init with the known seed yields reproducible weights.
3. **Construct a known batch.** 3–4 timesteps of hand-crafted `(states, actions, rewards, next_states, dones)`.
4. **Compute the expected loss by hand.** Trace the exact math — `discount()`, `center()`, log-probs from the seeded network, critic values — and document each intermediate step as inline comments.
5. **Assert `torch.allclose(agent.loss(*batch), expected, atol=1e-6)`.** Float32 precision, tight enough to catch math bugs.

The tests also serve as executable documentation of each algorithm's mathematical semantics.

### Agents and Their Loss Computations

| Agent | Loss formula | Batch shape |
|-------|-------------|-------------|
| VanillaPG | `mean(-log_π(a\|s) · G_t) + mean(-τ · H(π))` | 5 tensors: s, a, r, s', d |
| REINFORCE | VanillaPG + critic baseline: `mean(-log_π · (G_t - V(s)).detach()) + β·mean((G_t - V(s))²) + entropy` | 5 tensors |
| VanillaAC | One-step TD: `mean(-log_π · δ.detach()) + β·mean(δ²) + entropy` where `δ = r + γV(s') - V(s)` | 5 tensors |
| AdvantageAC | GAE: `mean(-log_π · A^GAE) + β·mean((A^GAE + V - V)²) + entropy` | 5 tensors |
| PPO | Clipped ratio: `-mean(min(r·A, clip(r)·A)) + β·mean((R - V)²) + entropy` | 8 tensors (includes old policy, advantages, returns) |
| VanillaDQN | `mean((r + γ·max Q_target(s') - Q(s,a))²)` | 5 tensors |

### SAM/LAMP Tests (separate module)

Test `Agent.learn()` rather than `loss()`:

1. **Vanilla vs SAM divergence.** Same agent, same batch. SAM-mode gradients differ from vanilla-mode gradients because the perturbation changes the loss landscape.
2. **Gradient shape preservation.** `get_grad`/`set_grad` round-trip maintains vector shape after the double forward pass.
3. **LAMP noise injection.** LAMP adds noise to the perturbation direction — compare with pure SAM on the same seed; they must differ.
4. **Parameter restoration.** After `learn()`, parameters moved from initial values (optimiser stepped) but the perturbation was fully undone (no lingering epsilon shift).

### Design Decisions

- **No mocking.** Real (tiny) networks test actual gradient flow.
- **`atol=1e-6`.** Tight enough for float32 to catch math bugs.
- **Intermediate values documented.** Each test has comments showing the hand computation so future maintainers can verify expected values.
- **One test per agent.** Each agent's loss is mathematically distinct — parametrization would obscure more than it clarifies.

---

## Card 9 — CI/CD with GitHub Actions

### Problem

No CI/CD pipeline exists. Pre-commit hooks run locally but nothing enforces quality on push or PR.

### Workflows

#### `ci.yml` — Push to main + all PRs

Single workflow with matrix strategy:

| Job | Command | Purpose |
|-----|---------|---------|
| lint | `ruff check rltrain/` + `ruff format --check rltrain/` | Style and import enforcement |
| typecheck | `pyright rltrain/` | Static type analysis (basic mode) |
| test | `pytest tests/ -v` | Full test suite |

All three jobs share the same setup:
- `actions/checkout@v4`
- `actions/setup-python@v5` with Python 3.10
- `astral-sh/setup-uv@v6` for uv
- `uv sync --group dev` to install deps

#### `release.yml` — Version tags (`v*`)

Triggered on tags matching `v*`:
1. Checkout at tag
2. Create GitHub Release with auto-generated release notes

No PyPI publish. Users install via `pip install git+...` until the project is ready for public release.

### Design Decisions

- **Single Python version (3.10).** Matches pyrightconfig.json minimum. Multi-version matrix deferred until public release.
- **No caching.** uv is fast enough that caching adds complexity for minimal gain.
- **Pre-commit stays local.** CI runs the same checks independently, not through pre-commit.
- **No PyPI.** GitHub Release only — public packaging deferred.

---

## Card 11 — Experiment Tracking Callback

### Problem

rltrain has no experiment tracking integration. The Callback Protocol was designed to enable this — training metrics go to CSV files, but researchers need WandB dashboards, TensorBoard, or custom trackers.

### Architecture

```
rltrain/tracking/
├── __init__.py          # Re-exports MetricsLogger, TrackingCallback
├── logger.py            # MetricsLogger protocol
├── callback.py          # TrackingCallback (Callback → Logger adapter)
└── backends/
    ├── __init__.py
    ├── stream.py         # StreamLogger (zero-dependency default)
    ├── fs.py             # FSLogger (JSONL to any fsspec filesystem)
    ├── tensorboard.py    # TensorBoardLogger
    ├── wandb.py          # WandbLogger
    └── xptrack.py        # XptrackLogger
```

### `MetricsLogger` Protocol

```python
@runtime_checkable
class MetricsLogger(Protocol):
    def start(self, config: dict, run_dir: Path) -> None: ...
    def log_scalars(self, metrics: dict[str, float], step: int) -> None: ...
    def log_hyperparams(self, params: dict[str, Any]) -> None: ...
    def finish(self) -> None: ...
```

Four methods. `start`/`finish` bracket the run. `log_scalars` handles all numeric metrics. `log_hyperparams` records the JSON config once at start.

### `TrackingCallback`

Thin adapter mapping Callback Protocol hooks to MetricsLogger calls. Constructed with a `MetricsLogger` instance and the experiment `config: dict` (the JSON config used to build the agent). The config is passed at construction because `Callback.on_train_start` receives `(agent, env, run_dir)` but not the raw config.

| Hook | Logger call |
|------|-------------|
| `on_train_start` | `logger.start(config, run_dir)` + `logger.log_hyperparams(config)` |
| `on_episode_end` | `logger.log_scalars({"return", "length", "running_return"}, episode)` |
| `on_train_end` | `logger.finish()` |
| `on_step` | no-op |
| `on_checkpoint` | no-op |

### Backend Adapters

Each implements `MetricsLogger`:

| Backend | Wraps | Dependency | Output format |
|---------|-------|------------|---------------|
| `StreamLogger` | Any `IO[str]` (stdout, stderr) | None — zero-dependency default | Human-readable lines |
| `FSLogger` | `fsspec` filesystem + path | `fsspec` | JSONL (one JSON object per `log_scalars` call) |
| `TensorBoardLogger` | `torch.utils.tensorboard.SummaryWriter` | None extra (ships with PyTorch) | TensorBoard events |
| `WandbLogger` | `wandb.init()`, `wandb.log()`, `wandb.finish()` | Optional, import guarded | WandB cloud |
| `XptrackLogger` | xptrack client | Optional, import guarded | xptrack store |

`StreamLogger` is the default when no backend is configured. Human-readable metrics to stdout with no setup.

`FSLogger` writes to any fsspec-compatible path (`file:///local/path`, `s3://bucket/prefix`, `gs://bucket/prefix`). Each `log_scalars` call appends one JSONL line — self-describing, append-friendly, and schema-flexible if new metrics are added later.

### JSON Config Integration

Backends are FQN-addressable:

```json
{
  "callbacks": [
    {
      "fqn": "rltrain.tracking.TrackingCallback",
      "logger": {
        "fqn": "rltrain.tracking.backends.WandbLogger",
        "project": "my-experiment"
      }
    }
  ]
}
```

### Dependencies

WandB and xptrack are optional — not added to core deps in `pyproject.toml`. TensorBoard comes with PyTorch. `StreamLogger` needs nothing. `fsspec` is optional but lightweight.

### Design Decisions

- **Layered architecture.** `MetricsLogger` protocol is reusable outside the training loop (evaluation scripts, sweeps, custom pipelines). The callback is one consumer of the logger, not the only one.
- **`StreamLogger` as default.** Zero-dependency, zero-config. Progressive disclosure: stream → FSLogger → TensorBoard → WandB/xptrack.
- **No `log_artifact`.** Deferred until someone needs it — YAGNI.
- **Optional deps import-guarded.** `ImportError` with install instructions, not silent failures.
