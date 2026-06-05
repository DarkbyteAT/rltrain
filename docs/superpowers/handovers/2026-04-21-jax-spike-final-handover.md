# JAX Spike Final Handover

Date: 2026-04-21

## Ground Yourself

```bash
cd /Users/ammar/Documents/Research/rltrain/.worktrees/spike-jax-port/
git branch   # spike/jax-port

# Activate the spike venv (not the main rltrain venv)
source .venv-spike/bin/activate
# Or install from scratch:
pip install -r requirements-spike.txt

# Run the test suite
pytest tests/spike/ -v

# Run a quick training example
python -m spike.examples.cartpole_dqn
python -m spike.examples.cartpole_ppo
python -m spike.examples.cartpole_sac
```

The spike lives entirely under `spike/` and `tests/spike/`. It shares the repo with the existing PyTorch `rltrain/` package but has no import dependencies on it. The top-level `CLAUDE.md`, `AGENTS.md`, `README.md`, and `CONTRIBUTING.md` describe the PyTorch codebase, not the spike.

## What Was Built

A self-contained JAX/Equinox deep RL framework proving that the rltrain agent contract ports cleanly from PyTorch mutation-based design to JAX pure-functional design. The spike validates the core hypothesis: static agent modules + external TrainState pytrees compose with `jax.jit`, `jax.vmap`, `jax.lax.scan`, and `jax.pmap` without adapter code.

### Agents (11)

| Agent | Family | File | State Type |
|-------|--------|------|------------|
| VanillaPG | On-policy PG | `agents/vanilla_pg.py` | `TrainState` |
| REINFORCE | On-policy PG | `agents/reinforce.py` | `TrainState` |
| VanillaAC | On-policy AC | `agents/vanilla_ac.py` | `TrainState` |
| AdvantageAC | On-policy AC | `agents/advantage_ac.py` | `TrainState` |
| PPO | On-policy AC | `agents/ppo.py` | `TrainState` |
| SPO | On-policy AC | `agents/spo.py` | `TrainState` |
| VanillaDQN | Off-policy Q | `agents/vanilla_dqn.py` | `DQNState` |
| DoubleDQN | Off-policy Q | `agents/double_dqn.py` | `DQNState` |
| DistributionalDQN | Off-policy Q | `agents/distributional_dqn.py` | `DQNState` |
| SAC (discrete) | Off-policy AC | `agents/sac.py` | `SACState` |
| SAC (continuous) | Off-policy AC | `agents/sac.py` | `SACState` |

On-policy agents share the `OnPolicyAgent` base class (`agent.py:221-293`), which provides default `init`, `learn`, and `act` implementations. Off-policy agents implement the `Agent` Protocol directly.

### Tests (193 test functions)

26 test files under `tests/spike/`, covering:
- Per-agent unit tests (loss shape, gradient flow, param updates, protocol conformance)
- Gradient isolation tests for multi-objective agents (SAC critic/actor independence)
- Buffer operations (add, sample, drain, circular overwrite, PER integration)
- Distribution stability (SquashedNormal vs distreqx NaN demonstration)
- Env wrappers (GymnaxEnv, GymnasiumFallback)
- Trainer integration (PythonLoop, ScanLoop)
- Scan-based training (on-policy and off-policy lax.scan convergence)
- End-to-end training (DQN CartPole to return >50 in 50K steps)
- Head modules (DiscreteHead, GaussianHead, SquashedGaussianHead, BetaHead, GammaHead)

### Trainer (3 loop strategies)

The Trainer (`spike/trainer/`) is decomposed into four files:
- `_trainer.py` (205 lines) -- orchestrator: config derivation, action-shape detection, loop dispatch
- `_loops.py` (534 lines) -- three `TrainingLoop` implementations
- `_carry.py` (63 lines) -- `TrainCarry`, `TrainConfig`, `StepOutput` dataclasses
- `__init__.py` -- public re-exports

Loop strategies:
- **PythonLoop** -- Python for-loop with JIT'd agent methods. Works with any env (gymnasium or gymnax). Fires callbacks inline.
- **ScanLoop** -- `jax.lax.scan` inner loop with Python checkpoint boundaries. Uses `jax.eval_shape` for zero-cost metrics shape discovery. Callbacks fire at segment boundaries.
- **PmapLoop** -- Multi-device parallel training via `jax.pmap`. Each device runs an independent scan loop. Metrics averaged across devices at checkpoint boundaries. Falls back to ScanLoop on single-device.

The Trainer auto-selects: gymnax envs get ScanLoop, gymnasium envs get PythonLoop. Users override via `loop=` kwarg.

### Callbacks

- `Callback` Protocol (`spike/callbacks/__init__.py`) -- 5-hook observer protocol (`on_train_start`, `on_step`, `on_episode_end`, `on_checkpoint`, `on_train_end`). All hooks receive Python values, not JAX arrays.
- `CSVLoggerCallback` -- episode metrics to CSV, flushed at checkpoints.
- `VideoRecorderCallback` -- MP4 eval rollouts via a separate gymnasium env at checkpoint boundaries. Requires moviepy.

### Other modules

- `spike/buffer.py` -- fixed-capacity ring buffer as a `chex.dataclass` pytree. Supports `add`, `sample` (uniform with IS weights for PER), and `drain` (on-policy). All operations are pure functions compatible with `lax.scan`.
- `spike/distributions.py` -- `SquashedNormal` with numerically stable `log_prob` using the identity `log(1 - tanh^2(x)) = 2(log2 - x - softplus(-2x))`. Replaces distreqx's `Transformed(Normal, Tanh)` which produces NaN at moderate feature magnitudes.
- `spike/heads.py` -- Action head modules: `DiscreteHead`, `GaussianHead`, `SquashedGaussianHead`, `BetaHead`, `GammaHead`. Each maps network features to a distribution.
- `spike/networks.py` -- `MLP` module (Equinox).
- `spike/math.py` -- `discount()` for GAE/returns, `center()` for whitening.
- `spike/transitions.py` -- `Transition` dataclass and `make_transition` factory.
- `spike/env.py` -- `GymnaxEnv` (pure-functional, JIT/vmap/scan-compatible) and `GymnasiumFallback` (Python-loop only). `EnvCapabilities` NamedTuple drives Trainer dispatch.

### Examples (11)

Under `spike/examples/`:
- `cartpole_dqn.py`, `cartpole_ppo.py`, `cartpole_sac.py` -- basic training scripts
- `cartpole_full_demo.py` -- full Trainer API with callbacks and CSV logging
- `cartpole_multi_env.py` -- gymnasium multi-env (note: docstring may be misleading, see tech debt)
- `pendulum_ppo.py`, `pendulum_sac.py` -- continuous control
- `showcase_jit_trace.py` -- demonstrates JIT compilation of agent methods
- `showcase_scan_benchmark.py` -- PythonLoop vs ScanLoop speed comparison (currently broken, see tech debt)
- `showcase_maml_ppo.py` -- MAML meta-learning with PPO inner loop
- `showcase_vmap_seed_sweep.py` -- vectorised seed sweep via `jax.vmap`

## Success Criteria Scorecard

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Port all 6 PyTorch agents to JAX | **PASS** | 11 agents implemented (6 original + DoubleDQN, DistributionalDQN, SAC discrete/continuous, SPO). Exceeds the original 6. |
| 2 | Pure-functional agent contract (init/learn/act) | **PASS** | `Agent` Protocol in `agent.py:71-100`. All agents satisfy it (tested via `isinstance(agent, Agent)` in protocol tests). Static module + external state pytree. |
| 3 | Compose with jit, vmap, scan | **PASS** | ScanLoop runs the full train loop inside `lax.scan`. `showcase_vmap_seed_sweep.py` vmaps across seeds. `showcase_jit_trace.py` demonstrates JIT. All agents JIT-compatible (tested). |
| 4 | Trainer with callback hooks | **PASS** | `Trainer` class with 5-hook `Callback` Protocol. Two built-in callbacks (CSV, Video). Three loop strategies (Python, Scan, Pmap). |
| 5 | Test suite with unit + integration + e2e | **PASS** | 193 tests across 26 files. Unit tests per agent, integration tests for scan training, e2e test for DQN CartPole convergence. |
| 6 | Support both gymnax and gymnasium envs | **PASS** | `GymnaxEnv` for pure-functional envs, `GymnasiumFallback` for classic gymnasium. Trainer auto-dispatches via `env.capabilities`. |
| 7 | Numerically stable continuous SAC | **PASS** | Custom `SquashedNormal` replaces distreqx. Test proves distreqx NaN, proves custom implementation is finite. SAC continuous trains on Pendulum. |
| 8 | No dependency on PyTorch rltrain | **PASS** | `spike/` has zero imports from `rltrain/`. Separate `requirements-spike.txt`. Can be extracted to a standalone package. |

## Architecture Summary

### Agent Protocol

The core contract is three methods on a static `eqx.Module`:

```
init(key) -> S           # construct initial training state
learn(state, batch, key) -> (state, metrics)  # one optimisation step
act(state, obs, key) -> action               # action selection
```

The agent module holds hyperparameters and network architecture (static at JIT trace time). All mutable state (parameters, optimizer state, target parameters) lives in a `TrainState` pytree that flows through `lax.scan` as the carry. This separation means JIT traces the agent once, and `jax.grad` composes through `learn` because the state is a pure pytree.

### State Types

- `TrainState` -- canonical state with `params`, `opt_state`, `target_params`. Used by all on-policy agents and DQN variants.
- `DQNState(TrainState)` -- adds `epsilon` for epsilon-greedy decay.
- `SACState` -- standalone state with separate `actor_params`, `critic_params`, `log_alpha`, three optimizer states, and `target_critic_params`. Does not extend `TrainState` because the field structure is fundamentally different.

### OnPolicyAgent Base Class

`OnPolicyAgent` (`agent.py:221-293`) provides shared `init`, `learn`, and `act` for policy gradient and actor-critic agents. Subclasses override `_loss(batch)` only. PPO and SPO override `learn` for their epoch loops but inherit `init` and `act`.

The `collect_size` ClassVar (default 256) tells the Trainer how many transitions to accumulate before calling `learn`. Off-policy agents default to 1 (via `getattr(agent, 'collect_size', 1)` in the Trainer).

### Trainer Decomposition

The Trainer owns configuration derivation and delegates the training loop:

```
Trainer.__init__()       # derives collect_size, buffer_capacity, action_shape, selects loop
Trainer.fit(key)         # builds initial carry, delegates to loop.run()
Trainer.make_initial_state(key)  # builds TrainCarry for checkpoint resume

TrainingLoop.run(agent, env, initial_carry, config, callbacks)  # strategy pattern
```

The `TrainCarry` chex dataclass carries `agent_state`, `env_state`, `buffer`, `step_count`, and `key` through `lax.scan`. `TrainConfig` is a frozen Python dataclass bundling loop-invariant hyperparameters. `StepOutput` accumulates per-step data (done, return, length, metrics, did_learn) for deferred callback dispatch at segment boundaries.

### TrainingLoop Protocol

```python
class TrainingLoop(Protocol):
    def run(self, agent, env, *, initial_carry, config, callbacks) -> Any: ...
```

Three implementations: PythonLoop (any env), ScanLoop (gymnax only), PmapLoop (multi-device). The Trainer selects based on `env.capabilities.scan_rollout`.

## Key Design Decisions

### Structural stop-gradient

SAC's three losses (critic, actor, alpha) are differentiated with respect to separate parameter groups. Stop-gradient is structural: parameters not passed as arguments to a loss function cannot receive gradients. No `jax.lax.stop_gradient` calls needed. This is cleaner than PyTorch's `detach()` pattern and impossible to forget.

### On-policy vs off-policy derived from collect_size

The Trainer never inspects an `on_policy` flag. Instead, `collect_size > 1` means on-policy (drain the buffer), `collect_size == 1` means off-policy (sample from the buffer). This is resolved at Python level (trace time), so JAX tracing sees only one branch. The distinction is emergent from the agent's hyperparameters, not a type hierarchy.

### jax.eval_shape for metrics discovery

ScanLoop uses `jax.eval_shape` to discover the metrics pytree structure from `agent.learn` without executing any FLOPs. This lets the loop pre-allocate zero-filled metrics for the `_skip_learn` branch inside `lax.cond`, ensuring both branches return identical pytree structures. Non-scalar metrics are warned and dropped.

### optax.partition deferred

The spike uses optax directly rather than porting the FQN builder / JSON config system. Agents construct their own optimizers. This was intentional -- the spike validates the pure-functional agent contract, not the configuration system. The FQN builder is orthogonal and can be layered on top.

### SquashedNormal over distreqx Transformed

distreqx's `Transformed(Normal, Tanh)` produces NaN `log_prob` at moderate feature magnitudes due to catastrophic cancellation in `log(1 - tanh^2(x))`. The custom `SquashedNormal` uses the algebraically equivalent `2(log2 - x - softplus(-2x))` which is stable for all `x`. A test (`test_sac.py:493`) demonstrates the distreqx failure to justify the custom implementation.

### eqx.partition / eqx.combine for param extraction

Every agent uses `eqx.partition(self, eqx.is_array)` to split the module into trainable parameters and static structure, then `eqx.combine(params, static)` to reconstruct a live module for forward passes. This is the Equinox idiom for "extract params, run optimizer, put params back" without mutation.

## What's Left / Tech Debt

### `_loops.py` at 534 lines

Contains three loop implementations, shared predicates, shape discovery, and the pure scan body `_train_step`. Should be split into separate files (`_python_loop.py`, `_scan_loop.py`, `_pmap_loop.py`, `_shared.py`) for production. Currently manageable but will grow as loop features are added.

### No input validation

Agents accept any kwargs without validation. No checks for negative gamma, invalid learning rates, mismatched network dimensions, etc. The Trainer doesn't validate that `num_steps % checkpoint_steps == 0` (it warns but truncates silently). Production code needs explicit validation at construction time.

### No config / FQN system

The spike has no JSON config system, no FQN builder, no CLI. Agents are constructed programmatically in Python scripts. This is appropriate for a spike but means the JAX port cannot yet be driven by the same JSON configs as PyTorch rltrain.

### VideoRecorderCallback captures agent at construction

The `VideoRecorderCallback` receives the agent's `act` function and state at checkpoint time, which works. However, the callback protocol receives `agent_state` but not the agent module itself at `on_checkpoint`. For video recording, the callback needs both the agent (for `act`) and the current state. The current design works around this by storing the agent reference at construction, but this is a protocol gap -- the agent reference is captured once and never updated.

### `cartpole_multi_env.py` docstring may be misleading

The example is about gymnasium multi-env training, but the docstring should be checked for accuracy against the actual implementation. May reference patterns from before the Trainer decomposition.

### `showcase_scan_benchmark.py` is broken

After the Trainer decomposition into `_trainer.py` / `_loops.py` / `_carry.py`, this showcase script references the old monolithic Trainer API. Needs updating to use the new `TrainingLoop` strategy pattern.

### No spike-specific documentation

Zero `.md` files under `spike/`. The top-level docs describe PyTorch rltrain. A `spike/README.md` with the architecture diagram, agent table, and "How to Add an Agent" guide is needed. (Task #5 is addressing this.)

### TrainState naming is misleading

`TrainState` is used by on-policy agents and DQN. SAC defines `SACState` from scratch. The name suggests it's the canonical state for all agents, but it's really "simple agent state." Consider renaming to `SimpleTrainState` or documenting that complex agents define their own state types.

### No off-policy base class

`OnPolicyAgent` provides shared `init`/`learn`/`act` for on-policy agents. Off-policy agents (VanillaDQN, SAC) reimplement the `eqx.partition`/`eqx.combine` dance, Polyak averaging, and target network init independently. Common patterns could be extracted.

## How to Continue

### Immediate (before merging spike to main)

1. Fix `showcase_scan_benchmark.py` (task #7)
2. Write `spike/README.md` (task #5)
3. Decide on package naming: does `spike/` become `rltrain_jax/` or replace `rltrain/`?

### Short-term (real JAX port)

1. Add the FQN builder / JSON config system for JAX agents
2. Add a CLI (`run.py` equivalent) that drives training from JSON configs
3. Port the experiment tracking system (`TrackingCallback` + `MetricsLogger` backends)
4. Add `CheckpointCallback` for JAX (serialize with `eqx.tree_serialise_leaves`)
5. Add input validation to agent constructors and Trainer

### Medium-term (production readiness)

1. Split `_loops.py` into separate files per loop strategy
2. Extract an `OffPolicyAgent` base class or shared helpers for the partition/combine/Polyak pattern
3. Add continuous-action on-policy agents (PPO with Gaussian policy)
4. Add CI/CD pipeline for the JAX test suite
5. Update top-level `CLAUDE.md`, `AGENTS.md`, `README.md` to cover the JAX backend
6. Benchmark against CleanRL / PureJaxRL baselines for correctness and speed validation
