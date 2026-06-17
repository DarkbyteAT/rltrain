"""Offline supervised TD-regression sweep for the Fourier-vs-Linear plasticity probe.

See ``docs/research_logs/2026-06-17-prereg-offline.md`` for the pre-registered
design, primary/secondary metrics, and statistical test.

What this script does (per (arm, seed)):
    1. Loads the fixed replay (``results/offline_probe/replay_50k.npz``).
    2. Builds a Q-network with the arm's backbone (ConvD2RLMLP or
       ConvFourierD2RLMLP, ``feature_dim=128``).
    3. Trains via supervised TD regression for ``outer_steps`` outer steps.
       Each outer step samples a uniform mini-batch of size 128 from the
       replay and runs ``inner_steps`` gradient updates on that batch.
       Target network synced via Polyak EMA after each outer step.
    4. Probes ``effective_rank`` and ``sign_entropy`` of the bottleneck
       feature vector (``feature_dim``-shaped, same site for both archs)
       every 500 outer steps on a held-out 256-sample probe set.
    5. Logs td-loss every 100 outer steps.
    6. Saves ``probes.csv`` and ``final_critic.eqx`` under
       ``results/offline_probe/<arm>/seed_<i>/``.

The whole training body is a single ``jax.lax.scan`` so the bulk of the work
happens in compiled XLA. The probe checkpoints are unrolled segments — same
pattern as ``rltrain/trainer/_loops.py``'s ScanLoop.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float, PRNGKeyArray

from rltrain.networks import ConvD2RLMLP, ConvFourierD2RLMLP


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmSpec:
    """Hyperparameters for one arm."""

    name: str
    arch: str  # "linear" | "fourier"
    inner_steps: int


ARMS: dict[str, ArmSpec] = {
    "linear_1x": ArmSpec(name="linear_1x", arch="linear", inner_steps=1),
    "fourier_1x": ArmSpec(name="fourier_1x", arch="fourier", inner_steps=1),
    "linear_Nx": ArmSpec(name="linear_Nx", arch="linear", inner_steps=8),
}


# ---------------------------------------------------------------------------
# Diagnostics — recomputed inline so they accept the (batch, feature_dim) probe
# ---------------------------------------------------------------------------


def _effective_rank(feats: Float[Array, "batch d"]) -> Array:
    """``exp`` of spectral entropy of centred features over the batch axis."""
    feats = feats - feats.mean(axis=0, keepdims=True)
    s = jnp.linalg.svd(feats, compute_uv=False)
    p = s / (s.sum() + 1e-12)
    return jnp.exp(-jnp.sum(p * jnp.log(p + 1e-12)))


def _sign_entropy(feats: Float[Array, "batch d"]) -> Array:
    """Mean per-unit sign entropy in ``[0, 1]``."""
    p = jnp.clip(jnp.mean((feats > 0).astype(jnp.float32), axis=0), 1e-6, 1 - 1e-6)
    return jnp.mean(-(p * jnp.log2(p) + (1 - p) * jnp.log2(1 - p)))


# ---------------------------------------------------------------------------
# Bottleneck probe site — shape-matched across archs
# ---------------------------------------------------------------------------


def _linear_bottleneck(critic: ConvD2RLMLP, obs_batch: Float[Array, "B H W C"]) -> Array:
    """Post-Conv->ReLU->Linear->ReLU projection features. Shape: (B, feature_dim)."""

    def one(obs):
        x_chw = jnp.transpose(obs, (2, 0, 1))
        f = jax.nn.relu(critic.conv(x_chw))
        z = jax.nn.relu(critic.projection(f.reshape(-1)))
        return z

    return jax.vmap(one)(obs_batch)


def _fourier_bottleneck(critic: ConvFourierD2RLMLP, obs_batch: Float[Array, "B H W C"]) -> Array:
    """Post-FourierBottleneck projection features. Shape: (B, feature_dim).

    We probe at the SAME location in shape-space as the linear arm: the
    feature_dim-sized output that feeds the D2RL MLP. Comparing rank/entropy
    on this fair site is what the pre-registration tests; the sin/cos concat
    pre-projection lives in 2*n_freqs space and isn't comparable.
    """

    def one(obs):
        x_chw = jnp.transpose(obs, (2, 0, 1))
        f = jax.nn.relu(critic.conv(x_chw))
        return critic.bottleneck(f.reshape(-1))

    return jax.vmap(one)(obs_batch)


def bottleneck_features(critic: Any, obs_batch: Array, arch: str) -> Array:
    if arch == "linear":
        return _linear_bottleneck(critic, obs_batch)
    if arch == "fourier":
        return _fourier_bottleneck(critic, obs_batch)
    raise ValueError(f"unknown arch: {arch}")


# ---------------------------------------------------------------------------
# Replay slice helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Replay:
    obs: Array
    action: Array
    reward: Array
    next_obs: Array
    done: Array

    @property
    def n(self) -> int:
        return int(self.obs.shape[0])


def load_replay(path: Path) -> Replay:
    arr = np.load(path)
    return Replay(
        obs=jnp.asarray(arr["obs"]),
        action=jnp.asarray(arr["action"]),
        reward=jnp.asarray(arr["reward"]),
        next_obs=jnp.asarray(arr["next_obs"]),
        done=jnp.asarray(arr["done"]),
    )


def sample_batch(replay: Replay, key: PRNGKeyArray, batch_size: int) -> dict[str, Array]:
    idx = jax.random.randint(key, (batch_size,), 0, replay.n)
    return {
        "obs": replay.obs[idx],
        "action": replay.action[idx],
        "reward": replay.reward[idx],
        "next_obs": replay.next_obs[idx],
        "done": replay.done[idx],
    }


# ---------------------------------------------------------------------------
# TD loss + gradient step
# ---------------------------------------------------------------------------


def critic_call(critic: Any, obs_batch: Array) -> Array:
    """Forward pass over a batch; returns Q values of shape ``(B, num_actions)``."""
    return jax.vmap(critic)(obs_batch)


def td_loss_fn(critic: Any, target_critic: Any, batch: dict[str, Array], gamma: float) -> Array:
    """Huber on Q(s, a) vs r + gamma * max_a' Q_target(s', a') * (1 - done)."""
    q_all = critic_call(critic, batch["obs"])  # (B, num_actions)
    q_sa = jnp.take_along_axis(q_all, batch["action"][:, None].astype(jnp.int32), axis=-1).squeeze(-1)
    q_next_all = critic_call(target_critic, batch["next_obs"])
    q_next_max = jnp.max(q_next_all, axis=-1)
    target = batch["reward"] + gamma * q_next_max * (1.0 - batch["done"].astype(jnp.float32))
    return jnp.mean(optax.huber_loss(q_sa, jax.lax.stop_gradient(target)))


def make_grad_step(arch: str, gamma: float):
    """JIT a single grad-step closure."""

    @eqx.filter_jit
    def step(critic, target_critic, opt_state, batch, optim_update):
        loss, grads = eqx.filter_value_and_grad(td_loss_fn)(critic, target_critic, batch, gamma)
        updates, new_opt_state = optim_update(grads, opt_state, critic)
        critic = eqx.apply_updates(critic, updates)
        return critic, new_opt_state, loss

    return step


# ---------------------------------------------------------------------------
# Polyak EMA target sync
# ---------------------------------------------------------------------------


def polyak_update(target_critic: Any, critic: Any, tau: float) -> Any:
    target_params, target_static = eqx.partition(target_critic, eqx.is_array)
    params, _static = eqx.partition(critic, eqx.is_array)
    new_target_params = jax.tree.map(
        lambda t, p: (1.0 - tau) * t + tau * p if t is not None else None,
        target_params,
        params,
    )
    return eqx.combine(new_target_params, target_static)


# ---------------------------------------------------------------------------
# Main per-run driver
# ---------------------------------------------------------------------------


def build_critic(arch: str, key: PRNGKeyArray) -> Any:
    """Build the critic. obs is HWC (10, 10, 4) for Breakout-MinAtar; num_actions = 3."""
    if arch == "linear":
        return ConvD2RLMLP(
            height=10,
            width=10,
            in_channels=4,
            out_size=3,
            conv_channels=16,
            conv_kernel=3,
            feature_dim=128,
            mlp_width=256,
            mlp_depth=4,
            key=key,
        )
    if arch == "fourier":
        return ConvFourierD2RLMLP(
            height=10,
            width=10,
            in_channels=4,
            out_size=3,
            conv_channels=16,
            conv_kernel=3,
            feature_dim=128,
            n_freqs=256,
            w0=1.0,
            mlp_width=256,
            mlp_depth=4,
            key=key,
        )
    raise ValueError(f"unknown arch: {arch}")


def run_one(
    arm: ArmSpec,
    seed: int,
    replay: Replay,
    out_dir: Path,
    *,
    outer_steps: int,
    batch_size: int,
    gamma: float,
    tau: float,
    lr: float,
    probe_every: int,
    loss_every: int,
    probe_set_size: int,
) -> dict[str, Any]:
    """Train one (arm, seed) configuration, log probes, save final critic."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[arm={arm.name} seed={seed}] start: outer_steps={outer_steps} inner_steps={arm.inner_steps}")
    t0 = time.perf_counter()

    key = jax.random.key(seed)
    k_init, k_probe, k_train = jax.random.split(key, 3)

    critic = build_critic(arm.arch, k_init)
    target_critic = critic  # eqx.Module instances are frozen pytrees; a fresh copy is just the same tree.

    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(critic, eqx.is_array))
    grad_step = make_grad_step(arm.arch, gamma)

    # Fixed held-out probe set — same for every probe call within a run.
    probe_idx = jax.random.choice(k_probe, replay.n, (probe_set_size,), replace=False)
    probe_obs = replay.obs[probe_idx]

    @eqx.filter_jit
    def measure_probes(critic):
        feats = bottleneck_features(critic, probe_obs, arm.arch)
        return _effective_rank(feats), _sign_entropy(feats)

    # CSV columns: step, td_loss (or NaN), effective_rank (or NaN), sign_entropy (or NaN)
    rows: list[tuple[int, float, float, float]] = []
    # Initial probe at step 0 for the AUC integration baseline.
    rank0, ent0 = measure_probes(critic)
    rows.append((0, float("nan"), float(rank0), float(ent0)))

    keys = jax.random.split(k_train, outer_steps)
    inner_steps = arm.inner_steps

    for step_i in range(outer_steps):
        k_sample = keys[step_i]
        batch = sample_batch(replay, k_sample, batch_size)
        # ``inner_steps`` gradient steps on the same batch for this outer step.
        loss = jnp.array(0.0)
        for _ in range(inner_steps):
            critic, opt_state, loss = grad_step(critic, target_critic, opt_state, batch, optim.update)
        # Polyak target sync after the outer step.
        target_critic = polyak_update(target_critic, critic, tau)

        outer = step_i + 1
        if outer % loss_every == 0:
            rows.append((outer, float(loss), float("nan"), float("nan")))
        if outer % probe_every == 0:
            r, e = measure_probes(critic)
            rows.append((outer, float("nan"), float(r), float(e)))

    t1 = time.perf_counter()
    duration = t1 - t0
    print(f"[arm={arm.name} seed={seed}] done in {duration:.1f}s ({outer_steps / max(duration, 1e-6):.0f} steps/s)")

    # Write probes.csv
    with (out_dir / "probes.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["step", "td_loss", "effective_rank", "sign_entropy"])
        for r in rows:
            w.writerow(r)
    # Save final critic
    eqx.tree_serialise_leaves(out_dir / "final_critic.eqx", critic)
    # Echo config
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "arm": arm.name,
                "arch": arm.arch,
                "seed": seed,
                "outer_steps": outer_steps,
                "inner_steps": arm.inner_steps,
                "batch_size": batch_size,
                "gamma": gamma,
                "tau": tau,
                "lr": lr,
                "probe_every": probe_every,
                "loss_every": loss_every,
                "probe_set_size": probe_set_size,
                "duration_sec": duration,
            },
            indent=2,
        )
    )
    return {"arm": arm.name, "seed": seed, "duration_sec": duration}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", type=Path, default=Path("results/offline_probe/replay_50k.npz"))
    p.add_argument("--out-root", type=Path, default=Path("results/offline_probe"))
    p.add_argument("--arms", nargs="+", choices=list(ARMS.keys()), default=list(ARMS.keys()))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--outer-steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--probe-every", type=int, default=500)
    p.add_argument("--loss-every", type=int, default=100)
    p.add_argument("--probe-set-size", type=int, default=256)
    p.add_argument(
        "--interleave",
        action="store_true",
        help="Cycle (seed, arm) rather than the default arm-major loop. Recommended for compile-cache amortisation.",
    )
    args = p.parse_args()

    print(f"Loading replay from {args.replay}")
    replay = load_replay(args.replay)
    print(f"  n={replay.n}, obs shape={replay.obs.shape}, dtype={replay.obs.dtype}")

    # Build the run order: by default arm-major. With --interleave, seed-major
    # cycles each seed across all arms so the first seed of each arch pays
    # compile cost early (per pre-reg).
    if args.interleave:
        run_order = [(s, a) for s in args.seeds for a in args.arms]
    else:
        run_order = [(s, a) for a in args.arms for s in args.seeds]

    summary: list[dict[str, Any]] = []
    for seed, arm_key in run_order:
        arm = ARMS[arm_key]
        out_dir = args.out_root / arm.name / f"seed_{seed}"
        if (out_dir / "probes.csv").exists() and (out_dir / "final_critic.eqx").exists():
            print(f"[arm={arm.name} seed={seed}] skipping — already complete")
            continue
        summary.append(
            run_one(
                arm,
                seed,
                replay,
                out_dir,
                outer_steps=args.outer_steps,
                batch_size=args.batch_size,
                gamma=args.gamma,
                tau=args.tau,
                lr=args.lr,
                probe_every=args.probe_every,
                loss_every=args.loss_every,
                probe_set_size=args.probe_set_size,
            )
        )

    print()
    print("Sweep complete.")
    print(f"Total wall-clock: {sum(s['duration_sec'] for s in summary):.1f}s across {len(summary)} runs")
    for s in summary:
        print(f"  arm={s['arm']:>10} seed={s['seed']}  duration={s['duration_sec']:.1f}s")


if __name__ == "__main__":
    main()
