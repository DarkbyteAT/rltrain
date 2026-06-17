"""Smoke test for jax-mlx-plugin on Apple Silicon.

Covers: device discovery, array placement, random.normal, SVD, scan, vmap.
Times a 1024x1024 matmul on MLX vs CPU.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp


def banner(msg: str) -> None:
    print(f"\n=== {msg} ===")


def main() -> None:
    banner("Devices")
    devs = jax.devices()
    print(f"jax.devices() -> {devs}")
    print(f"jax.default_backend() -> {jax.default_backend()}")
    if not any("mlx" in str(d).lower() or "metal" in str(d).lower() for d in devs):
        raise SystemExit(f"FAIL: no MLX/Metal device found. Got {devs}")

    banner("Array placement")
    x = jnp.zeros((4, 4))
    print(f"jnp.zeros((4,4)).device -> {x.device}")
    print(f"jnp.zeros((4,4)).devices() -> {x.devices()}")

    banner("random.normal")
    key = jax.random.key(0)
    rn = jax.random.normal(key, (1024, 1024))
    print(f"random.normal((1024,1024)) shape={rn.shape} dtype={rn.dtype}")
    print(f"  mean={float(jnp.mean(rn)):+.4f} std={float(jnp.std(rn)):.4f}")

    banner("linalg.svd (used by effective_rank)")
    m = jax.random.normal(jax.random.key(1), (64, 64))
    u, s, vt = jnp.linalg.svd(m)
    print(f"svd shapes: u={u.shape} s={s.shape} vt={vt.shape}")
    print(f"  top-3 singular values: {[float(v) for v in s[:3]]}")

    banner("lax.scan")

    def step(carry, x):
        return carry + x, carry * 2

    final, ys = jax.lax.scan(step, jnp.array(0.0), jnp.arange(5.0))
    print(f"scan final={float(final)} ys={[float(y) for y in ys]}")

    banner("vmap")

    def f(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(x**2)

    out = jax.vmap(f)(jnp.arange(12.0).reshape(3, 4))
    print(f"vmap(sum_sq) over (3,4) -> {[float(o) for o in out]}")

    banner("Matmul timing: MLX vs CPU")
    rng = jax.random.key(42)
    a = jax.random.normal(rng, (1024, 1024))
    b = jax.random.normal(jax.random.fold_in(rng, 1), (1024, 1024))

    # MLX (default backend)
    matmul_mlx = jax.jit(jnp.matmul)
    matmul_mlx(a, b).block_until_ready()  # warmup
    n_iter = 20
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out_mlx = matmul_mlx(a, b).block_until_ready()
    mlx_ms = (time.perf_counter() - t0) / n_iter * 1000
    print(f"MLX matmul 1024x1024 : {mlx_ms:7.2f} ms/iter (mean of {n_iter})")

    # CPU
    cpu = jax.devices("cpu")[0]
    a_cpu = jax.device_put(a, cpu)
    b_cpu = jax.device_put(b, cpu)
    matmul_cpu = jax.jit(jnp.matmul, device=cpu)
    matmul_cpu(a_cpu, b_cpu).block_until_ready()  # warmup
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out_cpu = matmul_cpu(a_cpu, b_cpu).block_until_ready()
    cpu_ms = (time.perf_counter() - t0) / n_iter * 1000
    print(f"CPU matmul 1024x1024 : {cpu_ms:7.2f} ms/iter (mean of {n_iter})")
    print(f"Speedup MLX vs CPU   : {cpu_ms / mlx_ms:6.2f}x")

    # Sanity: outputs roughly agree
    diff = float(jnp.max(jnp.abs(out_mlx - jax.device_put(out_cpu, devs[0]))))
    print(f"  max abs diff (MLX vs CPU output): {diff:.2e}")

    print("\nSMOKE PASS")


if __name__ == "__main__":
    main()
