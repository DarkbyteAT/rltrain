"""Dump 50k Breakout-MinAtar transitions under a random policy.

Random-action rollout means no learning loop is involved. We use the existing
``GymnaxEnv(num_envs=8)`` vectorisation + ``jax.lax.scan`` to fill the replay
on-device, then transfer the whole buffer to numpy and save as a single
``.npz`` archive: ``obs (N, H, W, C) int32``, ``action (N,) int32``,
``reward (N,) float32``, ``next_obs (N, H, W, C) int32``,
``done (N,) bool``.

The dump is the shared offline dataset that all three offline-probe arms train
on. Seed is fixed so the dataset is reproducible — every arm/seed reads the
same transitions.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rltrain.env import GymnaxEnv


def dump_replay(out_path: Path, total_transitions: int, num_envs: int, seed: int) -> None:
    """Roll out the random policy and save the resulting transitions as an .npz."""
    env = GymnaxEnv("Breakout-MinAtar", num_envs=num_envs)
    iters = total_transitions // num_envs
    iters = (iters // 100) * 100  # round so the scan body is fixed-shape friendly
    transitions_written = iters * num_envs
    print(f"Dumping {transitions_written} transitions ({iters} scan iters, num_envs={num_envs}).")

    key = jax.random.key(seed)
    k_reset, key = jax.random.split(key)
    state = env.reset(k_reset)

    def step(carry, _i):
        s, k = carry
        k_act, k_step, k_next = jax.random.split(k, 3)
        # Random policy: uniform over actions, vectorised over envs.
        actions = jax.random.randint(k_act, (num_envs,), 0, env.num_actions, dtype=jnp.int32)
        prev_obs = s.obs
        s_new = env.step(s, actions, k_step)
        transition = {
            "obs": prev_obs,
            "action": actions,
            "reward": s_new.reward,
            "next_obs": s_new.obs,
            "done": s_new.done,
        }
        return (s_new, k_next), transition

    t0 = time.perf_counter()
    (_state, _key), rollout = jax.lax.scan(step, (state, key), jnp.arange(iters))
    # rollout dict has arrays of shape (iters, num_envs, ...). Flatten (iters, num_envs) -> N.
    flat = jax.tree.map(lambda x: x.reshape(x.shape[0] * x.shape[1], *x.shape[2:]), rollout)
    flat = jax.tree.map(np.asarray, flat)
    t1 = time.perf_counter()
    print(f"Rollout + lift: {t1 - t0:.2f}s")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        obs=flat["obs"].astype(np.float32),
        action=flat["action"].astype(np.int32),
        reward=flat["reward"].astype(np.float32),
        next_obs=flat["next_obs"].astype(np.float32),
        done=flat["done"].astype(np.bool_),
    )
    print(f"Saved to {out_path} (size={out_path.stat().st_size / 1e6:.2f} MB)")
    # Quick sanity counters.
    print(
        f"  obs shape={flat['obs'].shape} dtype={flat['obs'].dtype}\n"
        f"  done rate={float(flat['done'].mean()):.4f} (~1 per episode)\n"
        f"  reward sum={float(flat['reward'].sum()):.2f}\n"
        f"  reward nonzero rate={float((flat['reward'] != 0).mean()):.4f}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("results/offline_probe/replay_50k.npz"))
    p.add_argument("--total", type=int, default=50_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    dump_replay(args.out, args.total, args.num_envs, args.seed)


if __name__ == "__main__":
    main()
