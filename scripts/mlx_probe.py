"""End-to-end probe: build ConvFourierD2RLMLP on MLX, forward pass, vmap over a batch.

Times forward + vmap forward on MLX vs CPU.
"""

from __future__ import annotations

import time

import equinox as eqx
import jax
import jax.numpy as jnp

from rltrain.networks import ConvFourierD2RLMLP


def banner(msg: str) -> None:
    print(f"\n=== {msg} ===")


def time_fn(fn, n: int = 20) -> float:
    fn().block_until_ready()  # warmup
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn().block_until_ready()
    return (time.perf_counter() - t0) / n * 1000


def main() -> None:
    banner("Devices")
    print(f"jax.devices() -> {jax.devices()}")
    print(f"default backend -> {jax.default_backend()}")

    banner("Build ConvFourierD2RLMLP")
    key = jax.random.key(0)
    net = ConvFourierD2RLMLP(
        height=10,
        width=10,
        in_channels=4,
        out_size=4,
        feature_dim=128,
        key=key,
    )
    print(f"Built net: {type(net).__name__}")
    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(net, eqx.is_array)))
    print(f"  param count: {n_params:,}")

    banner("Forward pass on (10, 10, 4) obs")
    obs = jnp.zeros((10, 10, 4), dtype=jnp.float32)
    out = net(obs)
    print(f"  out shape: {out.shape} dtype: {out.dtype}")
    print(f"  out[:4]: {[float(x) for x in out[:4]]}")

    banner("vmap over batch (256, 10, 10, 4)")
    batch = jax.random.normal(jax.random.key(1), (256, 10, 10, 4))
    batched = jax.vmap(net)(batch)
    print(f"  batched out shape: {batched.shape}")

    banner("Forward timing: MLX vs CPU")

    @eqx.filter_jit
    def single_fwd(net, x):
        return net(x)

    @eqx.filter_jit
    def batch_fwd(net, x):
        return jax.vmap(net)(x)

    mlx_single = time_fn(lambda: single_fwd(net, obs), n=20)
    mlx_batch = time_fn(lambda: batch_fwd(net, batch), n=20)
    print(f"MLX single forward  (10, 10, 4)      : {mlx_single:7.2f} ms/iter")
    print(f"MLX batch forward  (256, 10, 10, 4)  : {mlx_batch:7.2f} ms/iter")

    cpu = jax.devices("cpu")[0]
    net_cpu = jax.device_put(net, cpu)
    obs_cpu = jax.device_put(obs, cpu)
    batch_cpu = jax.device_put(batch, cpu)

    cpu_single = time_fn(lambda: single_fwd(net_cpu, obs_cpu), n=20)
    cpu_batch = time_fn(lambda: batch_fwd(net_cpu, batch_cpu), n=20)
    print(f"CPU single forward                   : {cpu_single:7.2f} ms/iter")
    print(f"CPU batch forward                    : {cpu_batch:7.2f} ms/iter")

    print(f"\nSpeedup single: {cpu_single / mlx_single:5.2f}x")
    print(f"Speedup batch : {cpu_batch / mlx_batch:5.2f}x")

    print("\nPROBE PASS")


if __name__ == "__main__":
    main()
