"""Train PPO on CartPole-v1 with multiple parallel environments.

Demonstrates the gymnasium pathway with ``num_envs > 1``. Each
``env.step()`` advances all ``NUM_ENVS`` copies and returns batched
observations of shape ``(N, obs_dim)``; ``PythonLoop`` dispatches to
``agent.act_batch`` on the hot path, so the agent picks ``N`` actions
per step from a single forward pass (vmapped under the hood).

Compare with ``cartpole_video_demo.py`` (single-env gymnax,
``lax.scan``-able) for the per-step-throughput vs broad-env-coverage
trade-off.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import jax

from rltrain.builders import agent as build_agent
from rltrain.callbacks.checkpoint import CheckpointCallback
from rltrain.callbacks.csv_logger import CSVLoggerCallback
from rltrain.callbacks.plot import PlotCallback
from rltrain.env import GymnasiumEnv
from rltrain.trainer import Trainer


EXAMPLES_DIR = Path(__file__).parent
AGENT_CFG = json.loads((EXAMPLES_DIR / "cartpole" / "ppo.json").read_text())
ENV_CFG = json.loads((EXAMPLES_DIR / "cartpole" / "env.json").read_text())
RUN_DIR = Path("results/cartpole_multi_env") / datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
NUM_STEPS = 500_000
CHECKPOINT_STEPS = 25_000
SEED = 42

# Number of parallel environment copies. Each env.step() advances all
# NUM_ENVS environments and returns NUM_ENVS transitions, so horizon
# buffers fill faster and the agent sees more diverse experience per
# update.
NUM_ENVS = 8


def main() -> None:
    """Run the multi-env CartPole PPO demo."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    key = jax.random.key(SEED)
    k_agent, k_fit = jax.random.split(key)

    # Inject num_envs into the agent config so PPO's per-env GAE reshape
    # bootstraps within each env's trajectory rather than across env
    # boundaries.
    agent_cfg = {**AGENT_CFG, "num_envs": NUM_ENVS}
    agent = build_agent(**agent_cfg, key=k_agent)
    # Force gymnasium backend; env.json's backend key is overridden inline
    # so a single shared cartpole/env.json can serve both demos.
    env = GymnasiumEnv(env_id=ENV_CFG["id"], num_envs=NUM_ENVS)

    trainer = Trainer(
        agent,
        env,
        num_steps=NUM_STEPS,
        checkpoint_steps=CHECKPOINT_STEPS,
        run_dir=RUN_DIR,
        callbacks=[CSVLoggerCallback(), PlotCallback(num_steps=NUM_STEPS), CheckpointCallback()],
        seed=SEED,
    )

    print(f"Training PPO on CartPole-v1 with {NUM_ENVS} parallel environments...")
    print(f"Each env.step() collects {NUM_ENVS} transitions ({NUM_ENVS}x data throughput)")
    trainer.fit(k_fit)
    print(f"Done. Results saved to {RUN_DIR}/")


if __name__ == "__main__":
    main()
