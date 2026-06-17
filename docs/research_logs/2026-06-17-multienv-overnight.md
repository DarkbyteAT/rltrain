# Multi-Env Overnight — Research Log

**Goal**: ship a vectorised `num_envs > 1` rltrain framework feature + run baseline ConvD2RLMLP and Fourier-variant ConvFourierD2RLMLP A/B on Space Invaders (MinAtar) until each meaningfully beats random baseline.

**Branch**: `feat/multi-env-d2rl-progression`
**Started**: 2026-06-17 (Ammar going to bed; autonomous run)

---

## Constraints & priors

- Single CPU machine. Wall-clock per single-env 100k MinAtar run was tracking ~60-90 min — too slow for the SpaceInvaders runs without vectorisation.
- gymnax MinAtar envs are JAX-pure and trivially `vmap`-able at the env level; the bottleneck is the framework, not the env (Trello card **SPR5Jy9R**).
- 100k env steps is a budget, not a solve. Discrete SAC on MinAtar literature targets 500k-1M for "beating" thresholds (Young & Tian 2019, Christodoulou 2019 follow-ups). With `num_envs=8` vectorisation, 500k env steps fits a single-night budget per run.
- Fixed-recipe experiment design: same SAC + PER + target_entropy=0.5*log(|A|) + 3e-4 Adam triple-optimiser across both bottleneck variants. Only the projection layer differs (Linear+ReLU vs frozen Fourier).

## Decisions taken before sleep

| Decision | Rationale |
|---|---|
| Stop the in-flight single-env Breakout baseline (was at 35% / ~500 episodes) | Single-env runs are sunk cost — every Space Invaders run wants vectorisation. Investment pays back across all subsequent experiments. |
| Re-mark 41 mis-categorised "unit" tests as integration | scan_on_policy.py + scan_off_policy.py compose lax.scan + agent.learn — clear integration territory by pytest.ini definition. per_integration.py has "integration" in its filename. Pre-commit unit gate is now 2:16 vs the prior 5:11. |
| Implement `num_envs > 1` via new `VectorisedScanLoop` (not modifying existing ScanLoop) | Lower risk overnight: existing ScanLoop unchanged, vectorised path is opt-in via env capabilities. Fall-back if framework work fails: single-env Space Invaders runs. |

## State at sleep handoff

- **Last committed**: `3ed4acd` (FourierBottleneck + ConvFourierD2RLMLP + re-marking).
- **In-flight commit**: runner + breakout_fourier config + RGB adapter for MinAtar video.
- **Run dirs cleaned**: removed empty `results/multi_env_progression/breakout_minatar/2026-06-16T23-31-45` (failed action_shape run) and stopped `2026-06-16T23-32-56` (single-env run we're abandoning). Will re-run after vectorisation lands.

## Plan (priority order)

1. ✅ Commit pending changes.
2. ⏳ Push branch.
3. ⏳ **Framework: num_envs > 1 for gymnax/ScanLoop path.**
   - `GymnaxEnv.__init__(num_envs)` — `vmap(reset)` and `vmap(step)` internally.
   - New `VectorisedScanLoop` cloned from ScanLoop with num_envs-leading-dim threading on obs/reward/done/episode_return/running_return.
   - Buffer change: `buffer_add_many(buffer, batched_transition)` adds num_envs transitions per scan step.
   - PER td_errors broadcast over envs.
   - Per-env episode callback firing in the host loop.
   - Tests: env-side reset/step shapes, scan-body shape contracts, end-to-end SAC at num_envs=4 over a small env.
4. ⏳ Baseline Breakout 100k with num_envs=8 — quick sanity ride. Expect ~5-15 min wall-clock.
5. ⏳ Fourier Breakout 100k with num_envs=8. Compare A/B via `compare_bottlenecks.py`.
6. ⏳ Baseline Space Invaders 400k+ with num_envs=8 — target "beats random" (return > 5 vs random ~0-2).
7. ⏳ Fourier Space Invaders 400k+ with num_envs=8.
8. ⏳ Final comparative report + PR open.

## Open questions / risks

- **Buffer write semantics under vmap**: `buffer.cursor` is per-buffer; adding N transitions in parallel needs sequential cursor advance. Likely path: scan over the num_envs axis inside buffer_add_many (sequential, but JAX-fused).
- **Episode callback fairness**: in vectorised mode, the host iterates `(num_envs, time)` boundaries. Need to preserve episode ordering (interleaved across envs is fine; per-env monotonic episode counters preferred for downstream analytics).
- **PER td_errors shape**: critic loss currently returns per-sample td_errors of shape `(B,)` where B = batch_size. With vectorised envs the batch is still drawn from the replay buffer per learn step; the td_errors shape doesn't change. Only the *write-back* path needs the per-env vmap when applying priority updates from the scan segment.
- **Action-shape detection**: still broken on image obs (Trello YNY6496u). The vectorised path is a natural place to do the proper fix via `env.num_envs`.

## Updates (chronological)

Entries appended below as work proceeds.

### 2026-06-17 ~01:05 — log seeded, framework work starting
- Re-marking landed. Unit suite 5:11 → 2:16.
- ConvFourierD2RLMLP A/B infrastructure committed (`3ed4acd`).
- About to start GymnaxEnv vectorisation.
