"""Train SAC with a D2RL dense-residual MLP on LunarLander-v3.

A fun visual demo showcasing rltrain's pluggable network surface: SAC learns
to land the lander using a D2RL-style dense-residual MLP (Sinha et al. 2020)
and three principled 3e-4 Adam optimisers (actor, twin critics, auto-tuned
alpha). Every hidden layer concatenates the raw observation with the previous
hidden activation; periodic video rollouts let you watch the agent improve
across checkpoints.

LunarLander-v3 is a gymnasium env, so the trainer auto-selects the Python
loop (not lax.scan). The agent itself still jits once; only the env step is
opaque to JAX.

Requires the Box2D physics backend, which is a demo-only soft requirement
and not part of rltrain's declared dependencies::

    uv pip install "gymnasium[box2d]"
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
AGENT_CFG = json.loads((EXAMPLES_DIR / "sac_d2rl.json").read_text())
ENV_CFG = json.loads((EXAMPLES_DIR / "env.json").read_text())
RUN_DIR = Path("results/lunarlander_d2rl") / datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
NUM_STEPS = 500_000
CHECKPOINT_STEPS = 25_000
SEED = 42


def main() -> None:
    """Run the LunarLander SAC + D2RL + video-recording demo."""
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
        batch_size=256,
        buffer_capacity=100_000,
        min_buffer_size=1_000,
        callbacks=[
            CSVLoggerCallback(),
            PlotCallback(num_steps=NUM_STEPS),
            CheckpointCallback(),
            VideoRecorderCallback(
                agent=agent,
                env_fn=lambda: gymnasium.make(ENV_CFG["id"], render_mode="rgb_array"),
                num_episodes=1,
            ),
        ],
        seed=SEED,
    )

    print(f"Training SAC + D2RL on {ENV_CFG['id']} for {NUM_STEPS:,} steps...")
    print(f"Run directory: {RUN_DIR}")
    print(f"Videos will be saved to {RUN_DIR / 'videos'}/")
    trainer.fit(k_fit)
    print(f"Done. Results saved to {RUN_DIR}/")


if __name__ == "__main__":
    main()
