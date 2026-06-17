"""Multi-env progression demo: fixed SAC+D2RL recipe across three gymnax envs.

Runs SAC with a D2RL dense-residual backbone (Sinha et al. 2020) and
prioritised experience replay on three discrete-action gymnax envs of
progressively richer observation structure:

1. **CartPole-v1** — 4-d vector obs, 2 actions. The vector-input anchor that
   verifies the D2RLMLP + discrete-SAC + PER recipe is functioning end to end.
2. **Breakout-MinAtar** — ``(10, 10, 4)`` image obs, 3 actions. First image
   env; exercises the principled :class:`ConvD2RLMLP` extension that inserts
   a learned ``Conv -> Linear`` projection in front of the dense-skip MLP so
   the D2RL concat operates on a ``feature_dim``-sized vector matching the
   original D2RL paper's vector-input regime.
3. **SpaceInvaders-MinAtar** — ``(10, 10, 6)`` image obs, 4 actions. Wider
   action space and richer channel structure than Breakout.

The recipe is held constant across envs (gamma, tau, target_entropy = 0.5 *
log(|A|), three 3e-4 Adam optimisers, PER, D2RL hidden width 256 / depth 4)
so that any progression in returns reflects the env-shape transfer, not
recipe tuning per env.

**100k is a budget, not "solved".** Discrete SAC on MinAtar typically
converges over 500k-1M steps; 100k tests whether the fixed recipe trends in
the right direction within a CI-friendly budget, not whether it reaches the
literature's asymptote. The comparative summary printed at the end reports
first-quartile vs. last-quartile mean return per env so the reader can see
the trend without misreading the budget as a convergence claim.

Usage::

    uv run python examples/multi_env_progression/run_progression.py
    uv run python examples/multi_env_progression/run_progression.py --envs cartpole
    uv run python examples/multi_env_progression/run_progression.py --num-steps 5000

Outputs land under ``results/multi_env_progression/<env_short_name>/<timestamp>/``.
No video recording: gymnax envs don't expose a gymnasium ``rgb_array``
renderer, so a video callback would require a second eval env per gymnax env
which is out of scope for this demo.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax

from rltrain.builders import agent as build_agent
from rltrain.builders import env as build_env
from rltrain.callbacks.checkpoint import CheckpointCallback
from rltrain.callbacks.csv_logger import CSVLoggerCallback
from rltrain.callbacks.plot import PlotCallback
from rltrain.trainer import Trainer


EXAMPLES_DIR = Path(__file__).parent

ENVS: dict[str, tuple[Path, Path, str]] = {
    "cartpole": (
        EXAMPLES_DIR / "cartpole" / "sac_d2rl.json",
        EXAMPLES_DIR / "cartpole" / "env.json",
        "CartPole-v1",
    ),
    "breakout": (
        EXAMPLES_DIR / "breakout_minatar" / "sac_convd2rl.json",
        EXAMPLES_DIR / "breakout_minatar" / "env.json",
        "Breakout-MinAtar",
    ),
    "breakout_fourier": (
        EXAMPLES_DIR / "breakout_minatar_fourier" / "sac_convfourier_d2rl.json",
        EXAMPLES_DIR / "breakout_minatar_fourier" / "env.json",
        "Breakout-MinAtar (Fourier)",
    ),
    "spaceinvaders": (
        EXAMPLES_DIR / "spaceinvaders_minatar" / "sac_convd2rl.json",
        EXAMPLES_DIR / "spaceinvaders_minatar" / "env.json",
        "SpaceInvaders-MinAtar",
    ),
    "spaceinvaders_fourier": (
        EXAMPLES_DIR / "spaceinvaders_minatar_fourier" / "sac_convfourier_d2rl.json",
        EXAMPLES_DIR / "spaceinvaders_minatar_fourier" / "env.json",
        "SpaceInvaders-MinAtar (Fourier)",
    ),
}
ENV_SHORT_NAMES: dict[str, str] = {
    "cartpole": "cartpole",
    "breakout": "breakout_minatar",
    "breakout_fourier": "breakout_minatar_fourier",
    "spaceinvaders": "spaceinvaders_minatar",
    "spaceinvaders_fourier": "spaceinvaders_minatar_fourier",
}

DEFAULT_NUM_STEPS = 100_000
DEFAULT_CHECKPOINT_STEPS = 5_000
DEFAULT_SEED = 42


class FlushingCSVLogger(CSVLoggerCallback):
    """CSV logger that flushes after every episode rather than at checkpoints.

    The stock :class:`CSVLoggerCallback` buffers episodes and flushes at
    ``on_checkpoint``. For watchable progress monitoring during a long
    multi-env run we want returns visible on disk the moment an episode
    finishes, so we override ``on_episode_end`` to write straight through.
    """

    def on_train_start(self, config: dict, run_dir: Path | None) -> None:
        """Open metrics.csv and write the header; same as the parent."""
        super().on_train_start(config, run_dir)

    def on_episode_end(
        self,
        episode: int,
        episode_return: float,
        episode_length: int,
        running_return: float = 0.0,
    ) -> None:
        """Write the episode row and flush immediately."""
        if self._writer is not None:
            self._writer.writerow([episode, episode_return, episode_length, running_return])
            assert self._file is not None
            self._file.flush()
        else:
            # No run_dir attached — fall back to in-memory buffering.
            super().on_episode_end(episode, episode_return, episode_length, running_return)

    def on_checkpoint(self, step: int, agent_state: Any, run_dir: Path | None) -> None:
        """Flush any residual buffered rows (none in the streaming path)."""
        super().on_checkpoint(step, agent_state, run_dir)


def _quartile_means(returns: list[float]) -> tuple[float, float]:
    """Return the mean of the first quarter and the mean of the last quarter."""
    n = len(returns)
    if n == 0:
        return float("nan"), float("nan")
    q = max(1, n // 4)
    first = sum(returns[:q]) / q
    last = sum(returns[-q:]) / q
    return first, last


def _read_metrics_csv(metrics_path: Path) -> list[dict[str, float]]:
    """Load the per-episode metrics.csv emitted by FlushingCSVLogger."""
    rows: list[dict[str, float]] = []
    if not metrics_path.exists():
        return rows
    with metrics_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append(
                    {
                        "episode": float(r["episode"]),
                        "return": float(r["return"]),
                        "length": float(r["length"]),
                        "running_return": float(r["running_return"]),
                    }
                )
            except (ValueError, KeyError):
                continue
    return rows


def run_env(
    env_key: str,
    *,
    num_steps: int | None,
    env_steps: int | None,
    checkpoint_steps: int,
    seed: int,
    timestamp: str,
    progression_index: int,
    progression_total: int,
) -> dict[str, Any]:
    """Train SAC+D2RL on one env, returning a summary dict for the final table.

    Exactly one of ``num_steps`` (scan iterations) or ``env_steps`` (env
    transitions) must be supplied. When ``env_steps`` is given the runner
    divides by the env's ``num_envs`` so each vectorised path runs roughly
    the same number of env transitions regardless of vectorisation factor.
    """
    agent_cfg_path, env_cfg_path, env_id = ENVS[env_key]
    agent_cfg = json.loads(agent_cfg_path.read_text())
    env_cfg = json.loads(env_cfg_path.read_text())

    run_dir = Path("results/multi_env_progression") / ENV_SHORT_NAMES[env_key] / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    key = jax.random.key(seed)
    k_agent, k_fit = jax.random.split(key)

    agent = build_agent(**agent_cfg, key=k_agent)
    env = build_env(**env_cfg)

    # Resolve scan-iteration count: env_steps takes priority and is divided by
    # num_envs so the env-transition budget stays constant.
    num_envs = getattr(env, "num_envs", 1)
    if env_steps is not None:
        scan_iters = max(env_steps // num_envs, checkpoint_steps)
        # Round down to a multiple of checkpoint_steps so the trainer doesn't warn.
        scan_iters = (scan_iters // checkpoint_steps) * checkpoint_steps
        if scan_iters < checkpoint_steps:
            scan_iters = checkpoint_steps
    else:
        assert num_steps is not None
        scan_iters = num_steps

    obs_shape = env.obs_shape
    num_actions = env.num_actions
    banner = (
        f"=== ENV {progression_index}/{progression_total}: {env_id} "
        f"· obs {tuple(obs_shape)} · actions {num_actions} "
        f"· num_envs={num_envs} · scan_iters={scan_iters} (env_steps={scan_iters * num_envs}) ==="
    )
    print(banner)
    print(f"Run directory: {run_dir}")

    trainer = Trainer(
        agent,
        env,
        num_steps=scan_iters,
        checkpoint_steps=checkpoint_steps,
        run_dir=run_dir,
        batch_size=256,
        buffer_capacity=100_000,
        min_buffer_size=1_000,
        prioritised=True,
        callbacks=[
            FlushingCSVLogger(),
            PlotCallback(num_steps=scan_iters),
            CheckpointCallback(),
        ],
        seed=seed,
    )
    trainer.fit(k_fit)

    rows = _read_metrics_csv(run_dir / "metrics.csv")
    returns = [r["return"] for r in rows]
    first_q, last_q = _quartile_means(returns)
    final_running = rows[-1]["running_return"] if rows else float("nan")
    total_length = int(sum(r["length"] for r in rows))

    print(f"--- {env_id}: {len(rows)} episodes, env_steps~{total_length}, last-quartile mean = {last_q:.2f}\n")

    return {
        "env_key": env_key,
        "env_id": env_id,
        "num_episodes": len(rows),
        "first_quartile_mean": first_q,
        "last_quartile_mean": last_q,
        "delta": last_q - first_q,
        "final_running_return": final_running,
        "total_env_steps": total_length,
        "run_dir": str(run_dir),
    }


def print_summary(summaries: list[dict[str, Any]]) -> None:
    """Print a comparative table across envs."""
    print()
    print("=" * 96)
    print("Multi-env D2RL progression summary")
    print("=" * 96)
    header = f"{'env':<26}{'episodes':>10}{'env_steps':>12}{'Q1 mean':>12}{'Q4 mean':>12}{'delta':>12}{'final_run':>12}"
    print(header)
    print("-" * 96)
    for s in summaries:
        print(
            f"{s['env_id']:<26}"
            f"{s['num_episodes']:>10d}"
            f"{s['total_env_steps']:>12d}"
            f"{s['first_quartile_mean']:>12.2f}"
            f"{s['last_quartile_mean']:>12.2f}"
            f"{s['delta']:>12.2f}"
            f"{s['final_running_return']:>12.2f}"
        )
    print("=" * 96)
    print("(100k steps is a CI-friendly budget, not a convergence target.)")


def main() -> None:
    """CLI entry-point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--envs",
        nargs="+",
        choices=list(ENVS.keys()),
        default=list(ENVS.keys()),
        help="Subset of envs to run, in order. Default: all three.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help=(
            "Scan-iteration count per env (raw Trainer.num_steps). With "
            "num_envs>1 each iter yields num_envs transitions. Use "
            "--env-steps to specify a transition budget instead."
        ),
    )
    parser.add_argument(
        "--env-steps",
        type=int,
        default=None,
        help=(
            "Env transition budget per env. Divided by num_envs to compute "
            "scan iterations. Mutually exclusive with --num-steps."
        ),
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        default=DEFAULT_CHECKPOINT_STEPS,
        help=f"Scan iterations between checkpoints (default {DEFAULT_CHECKPOINT_STEPS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed reused across envs (default {DEFAULT_SEED}).",
    )
    args = parser.parse_args()

    if args.num_steps is not None and args.env_steps is not None:
        raise SystemExit("Specify at most one of --num-steps and --env-steps.")
    if args.num_steps is None and args.env_steps is None:
        # Backward compat: treat the default as raw scan iterations.
        args.num_steps = DEFAULT_NUM_STEPS

    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
    summaries: list[dict[str, Any]] = []
    for i, env_key in enumerate(args.envs, start=1):
        summary = run_env(
            env_key,
            num_steps=args.num_steps,
            env_steps=args.env_steps,
            checkpoint_steps=args.checkpoint_steps,
            seed=args.seed,
            timestamp=timestamp,
            progression_index=i,
            progression_total=len(args.envs),
        )
        summaries.append(summary)

    print_summary(summaries)


if __name__ == "__main__":
    main()
