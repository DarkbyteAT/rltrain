"""Live SAC training with the plasticity probe — Linear vs Fourier arms.

The offline TD-regression sweep (``examples/offline_plasticity_probe/``) tests
whether the Fourier bottleneck preserves rank/entropy better than the Linear
bottleneck under matched supervised conditions. If it does, this script lights
up the live-RL follow-up: same probes, but inside a real SAC training loop.

Two arms supported via ``--arch``:

- ``linear`` — :class:`rltrain.networks.ConvD2RLMLP` actor + twin critics.
- ``fourier`` — :class:`rltrain.networks.ConvFourierD2RLMLP` actor + twin critics.

Both use the same SAC hyperparameters (target_entropy, tau, lr triples, batch
size, buffer config). The Fourier arm's only architectural difference is the
frozen sin/cos basis swapped in for the learned linear projection.

The :class:`PlasticityProbeCallback` writes ``probes.csv`` (``step,
effective_rank, sign_entropy``) at each checkpoint, sourcing observations from
a frozen random-policy obs batch collected before training begins.

Usage::

    python examples/live_sac_plasticity/run_live_plasticity.py \\
        --arch fourier --seed 0 --steps 200_000 --probe-cadence 5000

Smoke (2k-step run per arch)::

    python examples/live_sac_plasticity/run_live_plasticity.py \\
        --arch linear --seed 0 --steps 2000 --probe-cadence 500 --num-envs 8
    python examples/live_sac_plasticity/run_live_plasticity.py \\
        --arch fourier --seed 0 --steps 2000 --probe-cadence 500 --num-envs 8
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax

from examples.live_sac_plasticity.minatar_rgb_adapter import _GymnaxMinAtarRGBAdapter
from examples.live_sac_plasticity.plasticity_probe import (
    PlasticityProbeCallback,
    fixed_random_obs_provider,
)
from rltrain.builders import agent as build_agent
from rltrain.builders import env as build_env
from rltrain.callbacks.checkpoint import CheckpointCallback
from rltrain.callbacks.csv_logger import CSVLoggerCallback
from rltrain.callbacks.plot import PlotCallback
from rltrain.callbacks.video_recorder import VideoRecorderCallback
from rltrain.trainer import Trainer


logger = logging.getLogger(__name__)

EXAMPLES_DIR = Path(__file__).parent

ARCH_CONFIGS: dict[str, Path] = {
    "linear": EXAMPLES_DIR / "sac_convd2rl_breakout.json",
    "fourier": EXAMPLES_DIR / "sac_convfourier_breakout.json",
}
ENV_CONFIG_PATH = EXAMPLES_DIR / "env.json"


class FlushingCSVLogger(CSVLoggerCallback):
    """CSV logger that flushes after every episode.

    Same rationale as the sibling in ``run_progression.py``: long runs want
    live visibility into return progression without waiting for a checkpoint.
    """

    def on_episode_end(
        self,
        episode: int,
        episode_return: float,
        episode_length: int,
        running_return: float = 0.0,
    ) -> None:
        """Write the row and flush immediately."""
        if self._writer is not None:
            self._writer.writerow([episode, episode_return, episode_length, running_return])
            assert self._file is not None
            self._file.flush()
        else:
            super().on_episode_end(episode, episode_return, episode_length, running_return)


def run(
    *,
    arch: str,
    seed: int,
    steps: int,
    probe_cadence: int,
    num_envs: int,
    checkpoint_steps: int,
) -> dict[str, Any]:
    """Train one (arch, seed) configuration with the plasticity probe wired in."""
    if arch not in ARCH_CONFIGS:
        raise ValueError(f"unknown arch={arch!r}; expected one of {list(ARCH_CONFIGS)}")

    agent_cfg = json.loads(ARCH_CONFIGS[arch].read_text())
    env_cfg = json.loads(ENV_CONFIG_PATH.read_text())
    env_cfg["num_envs"] = num_envs

    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = Path("results/live_sac_plasticity") / arch / f"seed_{seed}" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    key = jax.random.key(seed)
    k_agent, k_obs, k_fit = jax.random.split(key, 3)

    agent = build_agent(**agent_cfg, key=k_agent)
    train_env = build_env(**env_cfg)

    # Frozen obs batch sourced from a random-policy rollout. Reusing the same
    # batch at every probe makes rank/entropy deltas attributable to network
    # changes rather than stimulus drift.
    obs_provider = fixed_random_obs_provider(train_env, k_obs, batch_size=256)

    callbacks = [
        FlushingCSVLogger(),
        PlotCallback(num_steps=steps),
        CheckpointCallback(),
        PlasticityProbeCallback(
            agent=agent,
            obs_provider=obs_provider,
            cadence_steps=probe_cadence,
        ),
        VideoRecorderCallback(
            agent=agent,
            env_fn=lambda: _GymnaxMinAtarRGBAdapter(env_id=env_cfg["id"]),
            num_episodes=1,
            max_steps=500,
            fps=15,
        ),
    ]

    print(
        f"=== live-SAC plasticity probe · arch={arch} · seed={seed} · "
        f"steps={steps} · probe_cadence={probe_cadence} · num_envs={num_envs} ===\n"
        f"run_dir: {run_dir}"
    )

    trainer = Trainer(
        agent,
        train_env,
        num_steps=steps,
        checkpoint_steps=checkpoint_steps,
        run_dir=run_dir,
        batch_size=128,
        buffer_capacity=100_000,
        min_buffer_size=1_000,
        prioritised=True,
        callbacks=callbacks,
        seed=seed,
    )
    trainer.fit(k_fit)

    probes_csv = run_dir / "probes.csv"
    n_probes = 0
    if probes_csv.exists():
        with probes_csv.open() as f:
            n_probes = sum(1 for _ in csv.reader(f)) - 1  # -1 for header
    metrics_csv = run_dir / "metrics.csv"
    n_episodes = 0
    if metrics_csv.exists():
        with metrics_csv.open() as f:
            n_episodes = sum(1 for _ in csv.reader(f)) - 1
    return {
        "arch": arch,
        "seed": seed,
        "run_dir": str(run_dir),
        "n_probes": n_probes,
        "n_episodes": n_episodes,
    }


def main() -> None:
    """CLI entry-point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=list(ARCH_CONFIGS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200_000, help="Scan iterations.")
    parser.add_argument(
        "--probe-cadence",
        type=int,
        default=5000,
        help="Probe at least every N steps (additionally fires at each checkpoint).",
    )
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--checkpoint-steps", type=int, default=500)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    summary = run(
        arch=args.arch,
        seed=args.seed,
        steps=args.steps,
        probe_cadence=args.probe_cadence,
        num_envs=args.num_envs,
        checkpoint_steps=args.checkpoint_steps,
    )

    print(
        f"--- arch={summary['arch']} seed={summary['seed']}: "
        f"{summary['n_episodes']} episodes, {summary['n_probes']} probe rows. "
        f"run_dir={summary['run_dir']}"
    )


if __name__ == "__main__":
    main()
