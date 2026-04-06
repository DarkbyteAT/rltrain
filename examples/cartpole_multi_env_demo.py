"""Train PPO-SAM on CartPole-v1 with multiple parallel environments.

Demonstrates multi-environment vectorisation: instead of stepping one environment
per call, we step NUM_ENVS copies simultaneously via gymnasium's SyncVectorEnv.
The agent receives a batch of observations and returns a batch of actions, collecting
NUM_ENVS transitions per step() call.

Compare with cartpole_video_demo.py (single-env) to see the difference in
wall-clock training speed.

Note: SyncVectorEnv steps all environments sequentially in a single process —
it provides more data per step() call but does not parallelise the env.step()
computation itself. For true multiprocessing parallelism, see AsyncVectorEnv
(tracked as a future enhancement).
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import rltrain.utils.builders as mk
from rltrain.callbacks.checkpoint import CheckpointCallback
from rltrain.callbacks.csv_logger import CSVLoggerCallback
from rltrain.callbacks.plot import PlotCallback
from rltrain.callbacks.video_recorder import VideoRecorderCallback
from rltrain.env import MDP
from rltrain.trainer import Trainer
from rltrain.utils.device import resolve_device


# --- Config ---
EXAMPLES_DIR = Path(__file__).parent
AGENT_CFG = json.loads((EXAMPLES_DIR / "cartpole" / "ppo.json").read_text())
ENV_CFG = json.loads((EXAMPLES_DIR / "cartpole" / "env.json").read_text())
RUN_DIR = Path("results/cartpole_multi_env") / datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
NUM_STEPS = 500_000
CHECKPOINT_STEPS = 25_000
SEED = 42

# Number of parallel environment copies.  Each step() call advances all NUM_ENVS
# environments and returns NUM_ENVS transitions.  MDP.total_steps increments by
# NUM_ENVS per call, so horizon buffers (e.g. PPO's 256-step horizon) fill faster
# and the agent learns from more diverse experience at each update.
NUM_ENVS = 8

# --- Build ---
agent = mk.agent(device=resolve_device("auto"), **AGENT_CFG)

# Pass num_envs to the env builder — this is the only change needed vs single-env.
# mk.env() creates a SyncVectorEnv wrapping NUM_ENVS independent copies of CartPole.
env = MDP(mk.env(num_envs=NUM_ENVS, **ENV_CFG), run_beta=0.05, log_freq=10, swap_channels=False)

# --- Train ---
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
            env_fn=lambda: mk.eval_env(**ENV_CFG),
            num_episodes=1,
        ),
    ],
    seed=SEED,
)

print(f"Training {agent.name} on CartPole-v1 with {NUM_ENVS} parallel environments...")
print(f"Each step() call collects {NUM_ENVS} transitions ({NUM_ENVS}x data throughput)")
print(f"Videos will be saved to {RUN_DIR / 'videos'}/")
trainer.fit()

print(f"\nDone! Final running return: {env.run_reward:.1f}")
print(f"Total episodes completed: {env.episode_count}")
print(f"Videos: {list((RUN_DIR / 'videos').glob('*.mp4'))}")
