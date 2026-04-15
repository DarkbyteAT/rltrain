r"""Full-featured CartPole training demo — mirrors PyTorch rltrain's cartpole_multi_env_demo.py.

Demonstrates the complete spike training pipeline:
- PPO agent with DiscreteHead on CartPole (gymnax, scan strategy)
- CSVLoggerCallback writing episode metrics to metrics.csv
- VideoRecorderCallback creating video directory at checkpoints
- Run directory with timestamped output

This is the "parity proof" — showing the JAX spike produces the same
artefacts (CSV metrics, video directory, trained state) as the PyTorch
version, with cleaner code and scan-level performance.
"""

from datetime import UTC, datetime
from pathlib import Path

import jax
import optax

from spike.agents.ppo import PPO
from spike.callbacks.csv_logger import CSVLoggerCallback
from spike.callbacks.video_recorder import VideoRecorderCallback
from spike.env import GymnaxEnv
from spike.heads import DiscreteHead
from spike.networks import MLP
from spike.trainer import Trainer


def main():  # noqa: D103
    key = jax.random.PRNGKey(42)
    k1, k2, k3, key = jax.random.split(key, 4)

    # --- Environment ---
    env = GymnaxEnv("CartPole-v1")

    # --- Agent ---
    agent = PPO(
        actor=MLP(env.obs_shape[0], 64, width=64, depth=1, key=k1),
        action_head=DiscreteHead(64, env.num_actions, key=k2),
        critic=MLP(env.obs_shape[0], 1, width=64, depth=1, key=k3),
        optimizer=optax.adam(3e-3),
        gamma=0.99,
        tau=0.01,
        beta_critic=0.5,
        lambda_gae=0.95,
        eps_clip=0.2,
        num_epochs=4,
        minibatch_size=64,
    )

    # --- Run directory ---
    run_dir = Path("results/spike_cartpole_ppo") / datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Callbacks (mirrors PyTorch demo) ---
    callbacks = [
        CSVLoggerCallback(),
        VideoRecorderCallback(),  # stub — creates directory structure
    ]

    # --- Train ---
    num_steps = 20_480
    checkpoint_steps = 5_120

    trainer = Trainer(
        agent,
        env,
        num_steps=num_steps,
        checkpoint_steps=checkpoint_steps,
        run_dir=run_dir,
        callbacks=callbacks,
        seed=42,
    )

    print("Training PPO on CartPole-v1 (gymnax scan strategy)")
    print(f"Run directory: {run_dir}")
    print(f"Steps: {num_steps}, checkpoints every {checkpoint_steps}")
    print()

    state = trainer.fit(key)

    # --- Verify outputs ---
    csv_path = run_dir / "metrics.csv"
    video_dir = run_dir / "videos"

    print(f"\n{'=' * 50}")
    print("Training complete. Output verification:")
    print(f"  CSV metrics:  {'FOUND' if csv_path.exists() else 'MISSING'} ({csv_path})")
    if csv_path.exists():
        lines = csv_path.read_text().strip().split("\n")
        print(f"    Header: {lines[0]}")
        print(f"    Rows:   {len(lines) - 1} episodes logged")
    print(f"  Video dir:    {'FOUND' if video_dir.exists() else 'MISSING'} ({video_dir})")
    print(f"  Agent state:  {type(state).__name__}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
