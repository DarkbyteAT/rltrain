# CLAUDE.md

Claude Code guidance for this repository.

@AGENTS.md

## Project Context

RLTrain is a PyTorch deep RL framework originally built for a 2022 dissertation (COMP3200, University of Southampton). It is being revived as an active research tool, open-source framework, and educational resource. It is a **meta-framework** — extensibility and clean abstractions are the product, not premature optimisation.

## Architecture at a Glance

- **Agent hierarchy**: `Agent` (ABC) → `VanillaPG` → `REINFORCE` → `VanillaAC` → `AdvantageAC` → `PPO`, plus `VanillaDQN`. Template method: `learn()` orchestrates, subclasses override `loss()` and `descend()`.
- **Environment**: `MDP` wraps `gymnasium.vector.SyncVectorEnv` — 5-tuple `step()` with `terminated | truncated` combined into `done` at the MDP level.
- **Trainer + Callbacks**: `Trainer.fit()` owns the loop. `Callback` Protocol with 5 hooks. Built-in: CSV, plots, checkpoints.
- **Configuration**: JSON files with `fqn` fields resolved by the FQN builder at runtime.
- **Gradient Transforms**: Composable `GradientTransform` pipeline (SAM, ASAM, LAMPRollback) applied between `loss.backward()` and `descend()`.
- **Experiment Tracking**: `TrackingCallback` adapts Callback hooks to pluggable `MetricsLogger` backends (Stream, JSONL, TensorBoard, W&B, xptrack).
- **Device**: `resolve_device("auto")` → CUDA → MPS → CPU. CLI: `--device {cpu,cuda,mps,auto}`.
- **Documentation**: MkDocs Material site at https://darkbyteat.github.io/rltrain.

## Known Technical Debt

- **Single-env vectorisation** — wraps 1 env in `SyncVectorEnv` (no true parallelism)
- **No CI/CD** pipeline

## Maintaining Documentation

Documentation is a first-class citizen. Update docs alongside code changes in the same PR.

| Audience | File | Contains | Does NOT contain |
|----------|------|----------|------------------|
| Users | `README.md` | Installation, usage, configuration, CLI args, examples | Dev conventions, architecture rules |
| Developers | `CONTRIBUTING.md` | Code conventions, testing, linting, architecture rules, key patterns | Usage guides, configuration reference |
| AI agents | `AGENTS.md` | Commands, rules, pitfalls, key doc references | Project context or architecture |
| Claude | `CLAUDE.md` (this file) | `@AGENTS.md` + architecture, design decisions, tech debt | Anything in README or CONTRIBUTING |

**When changing code:**
- New user-facing feature or API change → update `README.md`
- New convention, pattern, or architecture rule → update `CONTRIBUTING.md`
- New tech debt, resolved tech debt, or project-level context shift → update `CLAUDE.md`
- Never duplicate content across files — reference instead

**When creating PRs:** run `/gemini review`, resolve all comments, and re-run until convergence before requesting human review. See `CONTRIBUTING.md` for the full PR workflow.

**When resolving tech debt:** remove the entry from this file. When introducing tech debt: add it here with a brief explanation of why and what unblocks resolution.
