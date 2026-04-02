"""Train PPO on CartPole-v1 with video recording until it hits a score of 500."""

import json
from pathlib import Path

import gymnasium as gym
import torch as T

import rltrain.utils.builders as mk
from rltrain.callbacks.checkpoint import CheckpointCallback
from rltrain.callbacks.csv_logger import CSVLoggerCallback
from rltrain.callbacks.plot import PlotCallback
from rltrain.callbacks.video_recorder import VideoRecorderCallback
from rltrain.env import MDP
from rltrain.trainer import Trainer


# --- Config ---
EXAMPLES_DIR = Path(__file__).parent
AGENT_CFG = json.loads((EXAMPLES_DIR / "cartpole" / "ppo.json").read_text())
ENV_CFG = json.loads((EXAMPLES_DIR / "cartpole" / "env.json").read_text())
RUN_DIR = Path("results/cartpole_video_demo")
NUM_STEPS = 500_000
CHECKPOINT_STEPS = 25_000
SEED = 42

# --- Build ---
agent = mk.agent(device=T.device("cpu"), **AGENT_CFG)
env = MDP(mk.env(**ENV_CFG), run_beta=0.05, log_freq=10, swap_channels=False)

# --- Train with video recording ---
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
            env_fn=lambda: gym.make("CartPole-v1", render_mode="rgb_array"),
            num_episodes=1,
        ),
    ],
    seed=SEED,
)

print(f"Training {agent.name} on CartPole-v1 for {NUM_STEPS:,} steps...")
print(f"Videos will be saved to {RUN_DIR / 'videos'}/")
trainer.fit()

print(f"\nDone! Final running return: {env.run_reward:.1f}")
print(f"Videos: {list((RUN_DIR / 'videos').glob('*.mp4'))}")
