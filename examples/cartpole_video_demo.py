"""Train PPO on CartPole-v1 with periodic video recording.

Pure-JAX gymnax pathway: the agent jits once, the env is ``lax.scan``-able,
and ``VideoRecorderCallback`` periodically renders an evaluation episode
via a sibling gymnasium env for rgb_array support.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import gymnasium
import jax

from rltrain.builders import agent as build_agent
from rltrain.builders import env as build_env
from rltrain.callbacks.checkpoint import CheckpointCallback
from rltrain.callbacks.csv_logger import CSVLoggerCallback
from rltrain.callbacks.plot import PlotCallback
from rltrain.callbacks.video_recorder import VideoRecorderCallback
from rltrain.trainer import Trainer


EXAMPLES_DIR = Path(__file__).parent
AGENT_CFG = json.loads((EXAMPLES_DIR / "cartpole" / "ppo.json").read_text())
ENV_CFG = json.loads((EXAMPLES_DIR / "cartpole" / "env.json").read_text())
RUN_DIR = Path("results/cartpole_video_demo") / datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
NUM_STEPS = 500_000
CHECKPOINT_STEPS = 25_000
SEED = 42


def main() -> None:
    """Run the CartPole PPO + video-recording demo."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    key = jax.random.key(SEED)
    k_agent, k_fit = jax.random.split(key)

    agent = build_agent(**AGENT_CFG, key=k_agent)
    env = build_env(**ENV_CFG)

    trainer = Trainer(
        agent,
        env,
        num_steps=NUM_STEPS,
        checkpoint_steps=CHECKPOINT_STEPS,
        run_dir=RUN_DIR,
        callbacks=[
            CSVLoggerCallback(),
            PlotCallback(num_steps=NUM_STEPS),
            CheckpointCallback(),
            VideoRecorderCallback(
                env_fn=lambda: gymnasium.make(ENV_CFG["id"], render_mode="rgb_array"),
                num_episodes=1,
            ),
        ],
        seed=SEED,
    )

    print(f"Training PPO on CartPole-v1 for {NUM_STEPS:,} steps...")
    print(f"Videos will be saved to {RUN_DIR / 'videos'}/")
    trainer.fit(k_fit)
    print(f"Done. Results saved to {RUN_DIR}/")


if __name__ == "__main__":
    main()
