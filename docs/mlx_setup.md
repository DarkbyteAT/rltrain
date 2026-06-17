# JAX on Apple Silicon via `jax-mlx-plugin`

A reusable sidecar venv that exposes the Mac's GPU to JAX through Apple's MLX
runtime. Install once, then dispatch any python script with
`./scripts/run_mlx.sh <script>`.

## TL;DR — when MLX helps

`jax-mlx-plugin` 0.0.4 (on JAX 0.10) is **experimental**. From the
benchmarks below (see `## Smoke benchmarks`), the workload shapes that
benefit are:

- Large dense matmuls (`N ≥ 2048` square gives 6.7x on M-series CPU).
- Single large batched forward passes (B ≥ 4096 on a ~660k-param conv net
  gives ~1.5x).
- Large-batch gradient computations done one at a time outside `lax.scan`.

The workload shapes that **lose** under MLX:

- Per-step `lax.scan` over small forward passes — 9x slower than CPU. Each
  scan iteration eats a fresh kernel launch.
- `lax.scan` over `(grad → opt.update → apply_updates)` at small B — 1.8x
  slower than CPU on the `examples/offline_plasticity_probe` shape
  (B=128, scan over 100 outer steps).
- Single-sample forward passes — 7.6x slower than CPU (launch overhead
  dominates the µs-scale CPU dispatch).

In short: use MLX when arithmetic dominates dispatch. Bail to CPU when
the workload is a fine-grained scan over small kernels.

## Install

The sidecar lives at `.venv-mlx/`, separate from the main `.venv` (which
runs Python 3.12 + the main rltrain deps). `jax-mlx-plugin` ships only
Python-3.13 wheels, hence the split.

```bash
# Bootstrap (one-time)
uv python install 3.13                          # if not already present
./scripts/run_mlx.sh -c "import jax; print(jax.devices())"
# First run creates .venv-mlx, installs jax-mlx-plugin and rltrain editable.
# Expected output: [mlx:0]
```

The bootstrap is idempotent — `run_mlx.sh` skips it if `.venv-mlx/bin/python`
already exists.

If you prefer to do it manually:

```bash
PY313="$(uv python find 3.13)"
"$PY313" -m venv .venv-mlx
.venv-mlx/bin/pip install --upgrade pip
.venv-mlx/bin/pip install jax-mlx-plugin
.venv-mlx/bin/pip install -e .
```

That installs `jax==0.10.1`, `jaxlib==0.10.1`, `jax-mlx-plugin==0.0.4`,
`mlx==0.31.2`, `mlx-metal==0.31.2`, plus the rltrain JAX-ecosystem deps
(equinox, optax, distreqx, gymnax, jaxtyping, chex, flax, gymnasium).

## Usage

`run_mlx.sh` is the one-line dispatcher:

```bash
./scripts/run_mlx.sh path/to/script.py [args...]
./scripts/run_mlx.sh -m rltrain.cli --agent ... --env ...
```

It locates the worktree root from its own location, so it works from
any cwd and inside any worktree that has its own `.venv-mlx/`.

## Smoke and probe

Two scripts are committed for quick verification:

- `scripts/mlx_smoke.py` — device discovery, `random.normal`, `linalg.svd`,
  `lax.scan`, `vmap`, plus a 1024×1024 matmul timing.
- `scripts/mlx_probe.py` — builds `ConvFourierD2RLMLP` (the 660k-param
  conv-fourier net used in the offline plasticity probe), runs a single
  forward, vmaps over a batch of 256, and times forward on MLX vs CPU.

Run them:

```bash
./scripts/run_mlx.sh scripts/mlx_smoke.py
./scripts/run_mlx.sh scripts/mlx_probe.py
```

## Smoke benchmarks (M-series Mac, 2026-06-17)

### Matmul (`jnp.matmul`)

| N    | MLX (ms) | CPU (ms) | Speedup |
|------|----------|----------|---------|
| 1024 |     4.50 |     3.43 |   0.76x |
| 2048 |     7.92 |    53.25 |   6.72x |
| 4096 |   155.79 |   617.83 |   3.97x |

Crossover is around N ≈ 1500. Below that, CPU wins because MLX kernel
launches dominate.

### `ConvFourierD2RLMLP` forward, batched over B

| B    | MLX (ms) | CPU (ms) | Speedup |
|------|----------|----------|---------|
|   64 |     2.18 |     1.02 |   0.47x |
|  256 |     3.51 |     2.77 |   0.79x |
| 1024 |     6.96 |     6.85 |   0.98x |
| 4096 |    16.87 |    25.52 |   1.51x |

Crossover at B ≈ 1024. The offline plasticity probe uses B=128 → CPU is
faster.

### Realistic offline-sweep training body

`lax.scan` over 100 outer steps; each step does B=128 minibatch
sample → grad → `optax.adam` update on `ConvFourierD2RLMLP`:

| Backend | Time per scan |
|---------|---------------|
| MLX     |       1506 ms |
| CPU     |        830 ms |
| Speedup |         0.55x |

**MLX is 1.8x slower** for this exact workload. The 100 sequential
scan iterations each launch a separate MLX kernel; the per-launch
overhead exceeds the arithmetic savings at this batch size.

### Scan over 1000 single-step forwards (no batch dim)

| Backend | Time per scan |
|---------|---------------|
| MLX     |        672 ms |
| CPU     |         75 ms |
| Speedup |         0.11x |

The pathological case: 1000 sequential single-sample forwards. Don't
do this on MLX.

## Known issues

### Apple Silicon GPU only

The plugin loads `mlx-metal` and needs an M-series Mac. Intel Macs and
non-Mac platforms have no path here — they should fall back to the main
`.venv` (CPU JAX 0.4.x).

### Single-sample dispatch is slow

Per-call MLX kernel launch overhead is ~1.3ms even for a no-op forward
on a 660k-param net. Anything that calls the network in a tight Python
loop or in a small `lax.scan` will lose to CPU. Batch up, or stay on CPU.

### `jax-mlx-plugin` is experimental

The plugin emits `Platform 'mlx' is experimental and not all JAX
functionality may be correctly supported!` on every run. Operations
covered by the smoke test (`random.normal`, `linalg.svd`, `lax.scan`,
`vmap`, `eqx.nn.Conv2d` forward, `eqx.filter_grad` of an MSE loss,
`optax.adam` update) all run cleanly. Untested:

- `jax.pmap` (single device anyway on a Mac).
- `pure_callback` / `io_callback` (per memory entry
  `reference_jax_metal_macos.md`, `io_callback` is unsupported on both
  jax-mlx-plugin and jax-mps).
- `lax.cond` with closure-captured arrays (no failures seen, but not
  exhaustively probed).

If a workload hits an unsupported op, the failure mode is usually a
`NotImplementedError` from the MLX backend — not silent miscomputation.

### Output numerical agreement

The matmul smoke compares MLX vs CPU output bitwise and reports
`max abs diff = 0.00e+00` at N=1024. Forward passes through
`ConvFourierD2RLMLP` produce visually consistent outputs across backends
but were not exhaustively diffed.

## Recipe for re-bootstrapping (future Ammar / future Claude)

If `.venv-mlx/` is corrupted or you want to upgrade the plugin:

```bash
rm -rf .venv-mlx
./scripts/run_mlx.sh -c "print('rebuilt')"
```

If `jax-mlx-plugin` ships a new version, edit `scripts/run_mlx.sh`'s
`pip install jax-mlx-plugin` line to pin (`==X.Y.Z`) and re-bootstrap.

## When to extend this

If `jax-mlx-plugin` ever ships a wheel for Python 3.12, fold the sidecar
into the main `.venv` and delete this file. Until then, the sidecar
is the cleanest separation.
