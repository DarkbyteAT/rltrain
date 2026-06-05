# Changelog

All notable changes to rltrain are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/) as scoped by [`STABILITY.md`](./STABILITY.md).

## v1.0.0 — 2026-05-28

The JAX migration. rltrain is now a pure-JAX framework built on Equinox + Optax + distreqx + chex. The PyTorch lineage is gone from `main`; downstream consumers should pin v0.x for the legacy stack.

### Added

- **Agents** — `VanillaPG`, `REINFORCE`, `VanillaAC`, `AdvantageAC`, `PPO`, `SPO`, `VanillaDQN`, `DoubleDQN`, `DistributionalDQN` (C51), and `SAC` (discrete + continuous). All implement the `Agent` Protocol (`init`, `learn`, `act`, `act_batch`) and compose with `jax.grad` / `vmap` / `lax.scan`.
- **Action heads** — `DiscreteHead`, `GaussianHead`, `SquashedGaussianHead`, `GammaHead`, `BetaHead`, `CategoricalAtomHead`. All satisfy the `Head` Protocol (callable returning a `distreqx` distribution plus `action_dim`); `CategoricalAtomHead` is the value-head exception used by C51.
- **Training loops** — `PythonLoop`, `ScanLoop`, `PmapLoop` behind a common `TrainingLoop` Protocol. The `Trainer` auto-selects based on `env.capabilities`; supply `loop=...` to override.
- **Prioritised Experience Replay (PER)** — opt-in via `Trainer(prioritised=True)`. Off-policy agents (`VanillaDQN`, `DoubleDQN`, `DistributionalDQN`, `SAC`) consume `is_weights` via `_loss_weighted` and emit per-sample `td_errors`; the trainer routes them back into `buffer.priorities` at segment boundaries. Per-sample arrays never leave device.
- **PPO epoch terminators** — `EpochTerminator` Protocol and `KLEarlyStop` implementation. PPO's `learn` runs a double `lax.scan` over epochs × minibatches with a `stopped`-flag mask; the terminator chain is baked into a closure at construction time so the scan body is independent of the static tuple's contents.
- **Multi-env vectorisation** — `Agent.act_batch` with a default `jax.vmap` implementation. `PythonLoop` dispatches to `act_batch` when the env returns batched observations.
- **Callbacks** — `CSVLoggerCallback`, `PlotCallback`, `CheckpointCallback`, `VideoRecorderCallback` (with `eval_trigger` predicate for episode-driven recording).
- **CLI** — `python -m rltrain.cli` with JSON config + FQN builder.
- **Examples** — JSON configs for CartPole and Acrobot (a2c, reinforce, ppo) plus four runnable demo scripts (`cartpole_video_demo`, `cartpole_multi_env_demo`, `video_demo`, `tutorial_ppo_cartpole`).
- **Test markers** — five-marker split (`unit`, `integration`, `e2e`, `benchmark`, `slow`) with default exclusion of `benchmark` and `slow`. `make test` runs the default set; `make test-slow` and `make test-benchmark` run the opt-in suites.
- **Documentation** — `STABILITY.md` (this changelog's companion), `CHANGELOG.md`, MkDocs site at `docs/`, `tests/README.md` documenting marker semantics.

### Changed

- **`MLP` constructor** — accepts `width_size` (matching `eqx.nn.MLP`). The previous `width` alias is gone.
- **`DiscreteHead`** — kwargs are `feature_dim` and `action_dim` (the README example previously used the wrong names; now aligned).
- **Agent contract** — adds `act_batch` to the `Agent` Protocol. Existing agents pick up the vmap default automatically.
- **PPO and SPO** — `learn` rewritten to use a double `lax.scan` over epochs and minibatches; the per-epoch PRNG-key shuffle now flows through `buffer_shuffle_into_minibatches`, the generalised pytree shuffle helper.
- **Buffer** — `Transition` gains `is_weights` and `indices` fields with sentinel defaults; `buffer_sample` populates them on the returned batch.
- **Tests** — `tests/test_e2e_training.py` is now a fast smoke suite (500-1000 step training, finite-state assertions). The convergence tests with their 20k-50k step counts moved to `tests/test_e2e_training_slow.py` (`@pytest.mark.slow`).
- **Benchmark** — `scripts/bench_ppo_compile.py` migrated to `tests/test_benchmark.py` (`@pytest.mark.benchmark`).

### Removed

- **PyTorch lineage** — the entire pre-v1.0 implementation. `nn.Module` agents, the `Trajectory` collection class, the gymnasium-only loop, the SAM/LAMP gradient transforms (those moved to [samgria](https://github.com/DarkbyteAT/samgria)), the experiment-tracking stack (moved to [xptrack](https://github.com/DarkbyteAT/xptrack)), and the toblox network primitives (now in [toblox](https://github.com/DarkbyteAT/toblox)).
- **SAM/LAMP variant configs** — 12 SAM/LAMP JSON configs (`examples/{cartpole,acrobot}/*-sam.json` and `*-lamp.json`). Will return when samgria's JAX port lands.
- **`scripts/bench_ppo_compile.py`** — replaced by `tests/test_benchmark.py`.

### Performance

- **PPO compile** — `lax.scan` refactor + closure-at-init terminator baking gave ~4.8x compile-time and ~5.7x wall-clock improvement on CartPole PPO (vs the pre-refactor Python-loop version). Benchmark: `pytest tests/test_benchmark.py -m benchmark -s`.

### Migration notes

For consumers coming from rltrain v0.x:

- Replace `agent.learn()` orchestration with `Trainer(agent, env, ...).fit(key)`.
- Replace PyTorch `nn.Module` agent definitions with `eqx.Module` subclasses of `OnPolicyAgent` (PG/AC family) or standalone modules implementing the `Agent` Protocol (DQN/SAC family).
- Replace per-agent gradient-transform configs with `Trainer(prioritised=True)` for PER; SAM/ASAM/LAMP move to samgria.
- Update JSON configs: `width` → `width_size`, `in_features` → `feature_dim`, `num_actions` → `action_dim` on heads.
