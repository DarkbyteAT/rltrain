# Handover — JAX Port: Operational Phase

## Ground yourself

You are continuing the JAX port of rltrain. The spike is complete and validated on branch `spike/jax-port`.

- **Working directory:** `/Users/ammar/Documents/Research/rltrain/.worktrees/spike-jax-port/`
- **Branch:** `spike/jax-port`
- **Venv:** `.venv-spike` (JAX-only, no PyTorch)
- **Tests:** `source .venv-spike/bin/activate && python -m pytest tests/spike/ --noconftest -q -k "not trains_cartpole and not e2e"`
- **Examples:** `source .venv-spike/bin/activate && PYTHONPATH=. python spike/examples/<name>.py`

Read these first:
1. `spike/README.md` — architecture overview, agent hierarchy, how to add agents
2. `docs/superpowers/handovers/2026-04-21-jax-spike-final-handover.md` — what was built, success criteria scorecard, tech debt
3. `docs/superpowers/specs/2026-04-14-trainer-decomposition-design.md` — Trainer architecture
4. `spike/agents/agent.py` — Agent Protocol, OnPolicyAgent, TrainState, gradient_step

## What exists

- 11 agents (VanillaPG, REINFORCE, VanillaAC, AdvantageAC, PPO, SPO, VanillaDQN, DoubleDQN, DistributionalDQN, SAC continuous, SAC discrete)
- 193 tests across 26 files
- Trainer package with TrainingLoop protocol (PythonLoop, ScanLoop, PmapLoop)
- Callback protocol with CSVLoggerCallback and VideoRecorderCallback
- ExperienceBuffer with PER and IS weights
- SquashedNormal distribution (numerically stable, replaces distreqx)
- 11 runnable examples including scan benchmark, vmap seed sweep, MAML-PPO, full demo with CSV + video

## What is missing

### 1. FQN builder / JSON config system

PyTorch rltrain uses `rltrain/utils/builders/` to resolve JSON configs into object graphs at runtime. Users specify `{"fqn": "rltrain.agents.actor_critic.PPO", "gamma": 0.99, ...}` and the builder constructs the agent, networks, optimizer, and gradient transforms.

The spike has no equivalent. All construction is programmatic Python.

**What to build:**
- `spike/utils/builders/load.py` — `load(fqn: str)` that dynamically imports a class by fully-qualified name
- `spike/utils/builders/resolve.py` — `resolve(cfg: dict)` that recursively resolves a JSON config tree (dicts with `"fqn"` keys become constructed objects)
- `spike/utils/builders/agent.py` — `agent(cfg: dict, key: PRNGKeyArray)` that builds an Agent from JSON config, resolving MLP, action head, optimizer via FQN
- `spike/utils/builders/env.py` — `env(cfg: dict)` that builds a GymnaxEnv or GymnasiumEnv from JSON config

**Reference:** `rltrain/utils/builders/` and `rltrain/utils/README.md`

### 2. CheckpointCallback

PyTorch rltrain's `CheckpointCallback` saves `model.state_dict()` at checkpoints and train end, with optional save-all-checkpoints mode.

The spike has no checkpoint saving.

**What to build:**
- `spike/callbacks/checkpoint.py` — `CheckpointCallback` that serialises agent state at checkpoints using `eqx.tree_serialise_leaves` / `eqx.tree_deserialise_leaves`
- Save to `run_dir/checkpoints/step-{N}.eqx` (or similar)
- Save the agent config JSON alongside weights (requires FQN builder to exist first)

**Reference:** `rltrain/callbacks/checkpoint.py`

### 3. load_agent

PyTorch rltrain's `load_agent(run_dir, checkpoint, device)` reconstructs a trained agent from a saved run directory (config JSON + state dict).

The spike has no equivalent.

**What to build:**
- `spike/utils/builders/checkpoint.py` — `load_agent(run_dir: Path, checkpoint: str | None = None)` that:
  1. Reads `run_dir/config/agent.json`
  2. Reconstructs the agent via the FQN builder
  3. Loads weights via `eqx.tree_deserialise_leaves`
  4. Returns the agent module + loaded TrainState

**Depends on:** FQN builder (#1) and CheckpointCallback (#2)

**Reference:** `rltrain/utils/builders/checkpoint.py`

### 4. Experiment tracking backends

PyTorch rltrain has `TrackingCallback` that adapts Callback hooks to a pluggable `MetricsLogger` protocol, with 5 backends: StreamLogger, FSLogger (JSONL), TensorBoardLogger, WandbLogger, XptrackLogger.

The spike has CSVLoggerCallback only.

**What to build:**
- `spike/tracking/logger.py` — `MetricsLogger` Protocol (`start`, `log_scalars`, `log_hyperparams`, `finish`)
- `spike/tracking/callback.py` — `TrackingCallback` that wraps a `MetricsLogger` and adapts the spike's Callback hooks
- `spike/tracking/backends/stream.py` — `StreamLogger` (console output)
- `spike/tracking/backends/fs.py` — `FSLogger` (JSONL via fsspec)
- `spike/tracking/backends/tensorboard.py` — `TensorBoardLogger`
- `spike/tracking/backends/wandb.py` — `WandbLogger`

**Note:** The spike's Callback protocol has different signatures from PyTorch's (`on_step(step, metrics_dict)` vs `on_step(agent, env, step)`). The TrackingCallback must adapt to the spike's signatures.

**Reference:** `rltrain/tracking/README.md`, `rltrain/tracking/callback.py`, `rltrain/tracking/backends/`

### 5. GradientTransform equivalence via optax

PyTorch rltrain has a `GradientTransform` protocol (SAM, ASAM, LAMPRollback) applied between `loss.backward()` and `descend()`. The spike has no equivalent abstraction.

In JAX, SAM is expressible as a custom `optax.GradientTransformation`. LAMP requires a moving-average state field in TrainState.

**What to build:**
- `spike/transforms/sam.py` — SAM as an `optax.GradientTransformation` (or use `optax.contrib.sam` if available)
- `spike/transforms/asam.py` — ASAM variant
- `spike/transforms/lamp.py` — LAMPRollback as a stateful transform (requires adding `lamp_state` to TrainState or a separate carry field)
- Tests proving SAM+LAMP compose correctly via `optax.chain` (the PyTorch version's composition is broken for pre-descent stacking — confirmed by cross-examination)

**Reference:** `samgria/` (external package), the cross-examination findings in the port-readiness team's verdicts

### 6. PlotCallback

PyTorch rltrain's `PlotCallback` renders per-episode and per-sample return SVG plots at each checkpoint.

The spike has no plot callback.

**What to build:**
- `spike/callbacks/plot.py` — `PlotCallback` that renders return-vs-episode and return-vs-step SVG plots using matplotlib
- Save to `run_dir/per_episode.svg` and `run_dir/per_sample.svg`

**Reference:** `rltrain/callbacks/plot.py`

### 7. CLI wrapper

PyTorch rltrain has `run.py` — a typer CLI that accepts `--agent`, `--env`, `--dump`, `--device`, `--seed`, etc.

The spike has no CLI.

**What to build:**
- `spike/run.py` (or `run_spike.py`) — CLI wrapper using typer that accepts JSON config paths and delegates to the Trainer

**Depends on:** FQN builder (#1)

**Reference:** `run.py`

### 8. VideoRecorderCallback auto-detection

PyTorch rltrain's VideoRecorderCallback auto-detects the eval env from the training MDP's `EnvSpec`. The spike's version requires an explicit `env_fn` and captures the agent at construction time (protocol gap — Callback receives `agent_state` at checkpoint, not the agent module).

**What to fix:**
- Either pass the agent to the Callback protocol's `on_checkpoint` signature, or keep the current pattern of capturing the agent at construction time and document it

**Reference:** `rltrain/callbacks/video_recorder.py`
