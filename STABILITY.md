# API Stability

rltrain's public surface is split into three tiers. The tier determines what kind of change is allowed in what release.

## Tiers

| Tier | Promise | Breaking changes |
|---|---|---|
| **Stable** | Stays compatible across minor versions; deprecation cycles before removal | Major version bump only |
| **Provisional** | Compatible across patch versions; may evolve in minor versions | Minor version bump, documented in CHANGELOG |
| **Research-grade** | No stability guarantee; iterate freely | Any release |

If a symbol isn't listed below, treat it as **research-grade** by default.

## Stable

The contract you can depend on for production training scripts and downstream packages.

- **`rltrain.agents.Agent` Protocol** — `init(key) → state`, `learn(state, batch, key) → (state, metrics)`, `act(state, obs, key) → action`. Every agent shipped from rltrain satisfies this protocol structurally; user-supplied agents that match the shape work too.
- **`rltrain.callbacks.Callback` Protocol** — the five hooks (`on_train_start`, `on_step`, `on_episode_end`, `on_checkpoint`, `on_train_end`) and their signatures.
- **`Trainer` public surface** — `Trainer(agent, env, *, num_steps, checkpoint_steps, callbacks=None, seed=42, ...)` and `trainer.fit(key)`. The keyword arguments that already ship default-valued (callbacks, seed, batch_size, etc.) stay opt-in.
- **`rltrain.transitions.Transition` core fields** — `obs`, `action`, `reward`, `next_obs`, `done`. Plus the existing default-valued `log_prob` and `value` slots.
- **Public CLI** — `python -m rltrain.cli --agent <json> --env <json> --dump <dir>` and the documented flag set in `README.md`. Flag renames or removals require a major bump.
- **`rltrain.builders.agent.agent(fqn, *, key, **kwargs)` and `rltrain.builders.env.env(id, *, backend, ...)`** — the FQN builder entry points.
- **`rltrain.networks.MLP`** constructor: `MLP(in_size, out_size, *, width_size, depth, key)`. (The earlier `width` alias is gone; see `CHANGELOG.md v1.0.0`.)
- **Built-in callbacks** — `CSVLoggerCallback`, `PlotCallback`, `CheckpointCallback`. Constructors are stable; the metrics they consume mirror the Callback Protocol.

## Provisional

Likely to remain but may evolve. Track the CHANGELOG.

- **`rltrain.heads.Head` Protocol** — the callable + `action_dim` contract. The concrete heads (`DiscreteHead`, `GaussianHead`, `SquashedGaussianHead`, `GammaHead`, `BetaHead`) satisfy it today; the contract itself may grow.
- **`Agent.act_batch(state, obs, key) → actions`** — the batched-act method added with multi-env support. Default implementation is `default_act_batch` (vmap over `act`). The signature is stable but may be promoted into the Stable Protocol once we're sure no fused override patterns warrant a different shape.
- **`rltrain.agents.ppo_terminators.EpochTerminator` Protocol** and `KLEarlyStop` — the contract is `should_stop(metrics) → Bool[Array, ""]`. New terminators may be added; the shape of the `metrics` dict (currently `{"approx_kl"}`) may grow.
- **`rltrain.env.Env` Protocol** — pins `capabilities: EnvCapabilities` plus duck-typed `reset` / `step`. The two shipped backends (gymnax, gymnasium) have intentionally different signatures behind the Protocol; the Protocol itself is provisional and may tighten in a future minor.
- **`rltrain.env.EnvCapabilities`** — `(pure_step, vmap_batch, scan_rollout)`. Fields may be added; existing fields stay.
- **PER fields on Transition (`is_weights`, `indices`)** — added in v1.0.0 to plumb prioritised replay through `learn`. Sentinel defaults (`1.0`, `0`) so on-policy agents ignore them. The names and meanings are stable; tighter typing (jaxtyping shape annotations) may land.
- **`Trainer(prioritised: bool = False)`** — enables PER end-to-end. The flag is stable; the per-agent `_loss_weighted` / `td_errors` plumbing it relies on is also Provisional.

## Research-grade

Anything not listed above. Specifically:

- **Agent internals** — `_loss`, `_loss_weighted`, `_td_errors`, `_per_sample_xent`, `_check_stop` closure on PPO, the `_partition_params` / `_statics` / `_reconstruct_*` helpers on SAC. These will change as algorithms get tuned.
- **Loop implementations** — `PythonLoop`, `ScanLoop`, `PmapLoop` internals. The `TrainingLoop` Protocol is itself Provisional but the loop bodies are research-grade.
- **Buffer internals** — `ExperienceBuffer.priorities`, `buffer_update_priorities`, `_apply_per_updates_segment`. `make_buffer` and `buffer_sample` are part of the Trainer contract and so are Provisional, but the priority-update helpers can change shape.
- **`StepOutput` per-step fields** — used inside the scan body; not exposed via Callback hooks.
- **All example scripts under `rltrain/examples/`** — they show how to wire things up at a point in time; they are not a public API.

## Mapping changes to version bumps

| Change | Required bump |
|---|---|
| Remove or rename a Stable symbol | Major |
| Change the type signature of a Stable function in a non-backwards-compatible way | Major |
| Add new optional kwargs to a Stable function | Patch |
| Add a new Stable Protocol method (requires every implementation to update) | Major |
| Add a new Provisional Protocol method with a default implementation | Minor |
| Change a Provisional Protocol method's signature | Minor + CHANGELOG entry |
| Rename a Research-grade internal | Any |

## Deprecation policy

For Stable symbols:

1. The new shape ships in version `N`.
2. The old shape is marked deprecated in `N` with `DeprecationWarning`.
3. The old shape is removed no earlier than `N+1` (major bump).

For Provisional symbols, the deprecation cycle is one minor release.
