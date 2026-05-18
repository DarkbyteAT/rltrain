# CLAUDE.md

Claude Code guidance for this repository.

@AGENTS.md

## Project Context

RLTrain is a **JAX** deep RL framework, originally built as a PyTorch dissertation project at Southampton (2022) and ported to JAX in 2026. It is being maintained as an active research tool, open-source framework, and educational resource. It is a **meta-framework** — extensibility and clean abstractions are the product, not premature optimisation.

The PyTorch lineage is gone from `main`; the JAX port absorbed all retained features (CLI, JSON builders, callbacks, KLEarlyStop) and split out research surfaces (SAM/ASAM/LAMP → samgria, experiment tracking → xptrack, networks → toblox).

## Architecture at a Glance

- **Agent contract**: a `runtime_checkable` `Protocol` with three methods — `init(key) → state`, `learn(state, batch, key) → (state, metrics)`, `act(state, obs, key) → action`. Agents are `eqx.Module`s with no array leaves; state lives in a `chex.dataclass` (`TrainState`, `DQNState`, `SACState`) threaded through `learn`.
- **On-policy chain**: `OnPolicyAgent` base (in `rltrain/agents/agent.py`) — subclasses override `_loss(...)`. Linear progression: VanillaPG → REINFORCE → VanillaAC → AdvantageAC → PPO; SPO peers with PPO.
- **Off-policy**: VanillaDQN/DoubleDQN/DistributionalDQN share `dqn_learn_step`. SAC is structurally closer to DQN (twin Q + targets) than to the PG chain.
- **Environment**: `EnvCapabilities` NamedTuple gates dispatch. `GymnaxEnv` (pure_step, vmap, scan) for jittable in-XLA rollouts; `GymnasiumEnv` (Python loop) for broad env coverage.
- **Trainer + Loops**: `Trainer.fit(key)` auto-detects loop strategy from `env.capabilities`. Three impls behind the `TrainingLoop` Protocol: `PythonLoop`, `ScanLoop`, `PmapLoop` (with single-device fallback to ScanLoop).
- **Callbacks**: 5-hook `Protocol`. Hooks fire **Python-side at segment boundaries** — `ScanLoop` accumulates `StepOutput` arrays inside `lax.scan` and dispatches at each `checkpoint_steps` segment. No `io_callback`. See trainer source for the design trade-off.
- **Buffer**: one `ExperienceBuffer` `chex.dataclass` covering rollout, horizon mini-batch, and replay regimes via configuration. PER is a buffer option (`prioritised=True`), not a type.
- **Configuration**: JSON files with `fqn` fields resolved by `rltrain.builders`. Sub-objects are recursively constructed and threaded into parent constructors (PRNG sub-keys auto-spliced).
- **Documentation**: MkDocs Material site at https://darkbyteat.github.io/rltrain.

## Known Technical Debt

- **PER end-to-end wiring** — `_loss_weighted` exists on VanillaDQN and accepts IS weights, but the default trainer loop calls `_loss` (uniform). Wiring priorities back into the buffer is plumbing, not a design regression.
- **`buffer_shuffle_into_minibatches` helper** — implemented but bypassed by PPO's inline epoch shuffle. Either wire it in or remove.
- **Gaussian head divergence** — `GaussianHead` clips `log_sigma ∈ [-20, 2]` (numerical stability win over the original PyTorch); documented but not aligned to a single canonical form.
- **No CI/CD release pipeline** — `release.yml` exists but is untested.

## Maintaining Documentation

Documentation is a first-class citizen. Update docs alongside code changes in the same PR.

| Audience | File | Contains | Does NOT contain |
|----------|------|----------|------------------|
| Users | `README.md` | Installation, usage, configuration, CLI, examples | Dev conventions, architecture rules |
| Developers | `CONTRIBUTING.md` | Code conventions, testing, linting, architecture rules, key patterns | Usage guides, configuration reference |
| AI agents | `AGENTS.md` | Commands, rules, pitfalls, key doc references | Project context or architecture |
| Claude | `CLAUDE.md` (this file) | `@AGENTS.md` + architecture, design decisions, tech debt | Anything in README or CONTRIBUTING |

**When changing code:**
- New user-facing feature or API change → update `README.md`
- New convention, pattern, or architecture rule → update `CONTRIBUTING.md`
- New tech debt, resolved tech debt, or project-level context shift → update `CLAUDE.md`
- Never duplicate content across files — reference instead

**When creating PRs:** run `/gemini review`, resolve all comments, and re-run until convergence before requesting human review.

**When resolving tech debt:** remove the entry from this file. When introducing it: add it here with a brief explanation.
