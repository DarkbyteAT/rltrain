# Pre-Registration — Offline Supervised TD-Regression on Breakout-MinAtar Replay

**Committed before the 15-run sweep launches.** This file exists to lock in the
hypothesis, statistical test, success criterion, and analysis pipeline ahead
of seeing any per-arm result. Any deviation from what's here must be flagged
explicitly as an exploratory follow-up, not as a reinterpretation of the
pre-registered outcome.

## Why this design

Live SAC on this stack collapsed at ~ep 1281 (autopsy upstream in
`2026-06-17-multienv-overnight.md`). Trying to A/B Fourier vs Linear inside
that collapse regime makes Fourier's mechanistic claim — that random Fourier
features preserve representational diversity under repeated weight updates —
unfalsifiable: every run's final return is dominated by whether and when the
collapse fires.

Cleanest isolation per the consultations:

- **Contrarian's MVP framing**: rip the RL loop out. Generate a fixed offline
  replay under a random policy. Train a supervised TD-regression critic on
  that replay. Compare bottleneck `effective_rank` and `sign_entropy`
  trajectories between the architectures. No exploration, no target drift
  from policy improvement, no replay distribution shift.
- **Staff-architect's confound-control**: more gradient steps per env
  transition isn't a feature unique to Fourier conditioning — it's a free
  axis the live-SAC vectorisation introduces. Add a third arm
  (`Linear-Nx`, N matched to the UTD multiplier from the failed live-SAC
  setting) so the primary comparison is variant-on-variant at matched update
  budget.

## Data

- **Fixture**: `results/offline_probe/replay_50k.npz` (49,600 transitions, dumped 2026-06-17 at seed 0, random uniform policy, num_envs=8 vectorised gymnax rollout). Compressed npz keys: `obs (N, 10, 10, 4) float32`, `action (N,) int32`, `reward (N,) float32`, `next_obs (N, 10, 10, 4) float32`, `done (N,) bool`.
- **Same replay for every arm and seed**. The arms differ only in critic
  architecture and updates-per-batch; the dataset is held constant.

## Arms

| Arm | Critic backbone | Grad steps / replay batch | Purpose |
|---|---|---|---|
| **Linear-1x** | `ConvD2RLMLP` | 1 | Naive baseline (descriptive). |
| **Fourier-1x** | `ConvFourierD2RLMLP` | 1 | The mechanistic intervention. |
| **Linear-Nx** | `ConvD2RLMLP` | 8 | Confound-control: matched update budget vs Fourier-1x's "extra capacity per batch" interpretation. |

`N = 8` matches the UTD multiplier in the live-SAC setting that collapsed.

## Training loop

- For each (arm, seed):
  - Build critic per arm (same `feature_dim=128`, `conv_channels=16`, `conv_kernel=3`, `mlp_width=256`, `mlp_depth=4`; Fourier arm adds `n_freqs=256`, `w0=1.0`).
  - Build target critic = deep copy of critic.
  - Optimizer: `optax.adam(3e-4)`.
  - For each of 20,000 outer steps:
    - Sample a uniform mini-batch of size 128 from the replay.
    - Inner Polyak-EMA target sync at `tau = 0.005` after the update.
    - For `1` (Linear-1x / Fourier-1x) or `8` (Linear-Nx) grad steps:
      - Compute targets `r + gamma * max_a' Q_target(s', a') * (1 - done)`, `gamma = 0.99`.
      - Huber loss between `Q(s, a)` and the target.
      - One optimizer step.
  - Probe every 500 outer steps: `effective_rank`, `sign_entropy` of the bottleneck features on a fixed held-out 256-sample probe set drawn from the same replay.
  - Probe every 100 outer steps: mean `td_loss` over the most recent batch.
  - Save per-run `probes.csv` columns: `step, td_loss, effective_rank, sign_entropy` (rank/entropy NaN on TD-loss-only rows, NaN-padded on probe-only rows; downstream analysis ffills within run).

## Seeds

5 seeds per arm: `[0, 1, 2, 3, 4]`. 15 runs total.

## Run order (compile-cache amortisation)

Per skeptic: each arch pays cold-compile once. Cycle seeds across arms so the
first seed of each architecture eats the compile early.

```
Linear-1x seed 0
Fourier-1x seed 0
Linear-Nx seed 0
Linear-1x seed 1
Fourier-1x seed 1
Linear-Nx seed 1
…
Linear-1x seed 4
Fourier-1x seed 4
Linear-Nx seed 4
```

## Hypotheses, tests, success criteria

### Primary (pre-registered, the publishable outcome)

- **H1 (mechanistic)**: at iso-update-budget, Fourier conditioning preserves
  bottleneck feature diversity better than dense linear conditioning.
- **Operational metric**: `AUC(effective_rank)` over the 40 probe checkpoints
  (every 500 outer steps × 20k outer steps).
- **Test**: paired Wilcoxon signed-rank across the 5 seeds, **Fourier-1x vs
  Linear-Nx**.
- **Success criterion**: p < 0.05 AND median Fourier-1x AUC > median
  Linear-Nx AUC.
- **Failure modes pre-registered as such**: p ≥ 0.05; or Fourier-1x AUC ≤
  Linear-Nx AUC. Both are reported as null/negative.

### Secondary (pre-registered, supporting but not decisive)

- **H2**: at iso-update-budget, Fourier conditioning yields lower final TD
  loss than dense linear conditioning.
- **Metric**: mean `td_loss` over the last 1k outer steps.
- **Test**: paired Wilcoxon across seeds, Fourier-1x vs Linear-Nx.
- **Success criterion**: p < 0.05 AND median Fourier-1x TD-loss < median
  Linear-Nx TD-loss.

### Tertiary (descriptive only — no test)

- Linear-1x metrics reported as reference for the naive Fourier-vs-Linear
  comparison the original design intended. Comparisons against Linear-1x are
  exploratory and reported with point estimates only.
- `sign_entropy` is reported for both arms as a secondary diagnostic; no test
  pre-registered.

### Sanity / Goodhart guard

- Compute Spearman ρ between within-seed `AUC(effective_rank)` and within-seed
  `final TD-loss improvement` (= mean TD loss in first 1k steps − mean TD loss
  in last 1k steps) across all 15 runs.
- If |ρ| < 0.3 (and the H1 paired Wilcoxon is significant), flag the primary
  result as Goodhart-prone: rank trajectory and learning are decoupled.
- This is not a stopping criterion. It's a caveat that ships with the
  reported result.

## What's NOT pre-registered

- Wall-clock comparisons.
- Per-step gradient norms / weight-norm trajectories.
- Anything from a fourth or fifth arm. If we want a third arm beyond
  Linear-1x / Fourier-1x / Linear-Nx, it's an exploratory follow-up.

## Outputs

- One run directory per (arm, seed): `results/offline_probe/<arm>/seed_<i>/`.
  - `probes.csv`
  - `final_critic.eqx`
  - `config.json` echoing the arm hyperparameters
- One consolidated analysis section appended to
  `2026-06-17-multienv-overnight.md` after the sweep completes, citing this
  file by date.

## Hard caps

- Total wall-clock for sweep + analysis: ≤ 4 hours (per team-lead brief).
- If the sweep doesn't finish, the partial result is reported as such; no
  re-running on a fresh seed pool to "complete" the design after seeing
  partial outcomes.
