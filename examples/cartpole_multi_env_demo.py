"""Train PPO on CartPole-v1 via the gymnasium backend.

Demonstrates the gymnasium pathway — broad env coverage, Python step
loop. Compare with ``cartpole_video_demo.py`` (gymnax backend,
``lax.scan``-able, faster per step) for the trade-off.

Known limitation — true multi-env vectorisation is not yet wired
through the JAX trainer. ``GymnasiumEnv(num_envs=N)`` would batch
observations to shape ``(N, obs_dim)``, but ``PythonLoop`` calls
``agent.act(state, obs, key)`` assuming the unbatched shape
``(obs_dim,)`` — wiring an ``agent.act_batch`` or auto-``vmap`` on the
hot path is tracked as a future enhancement. For now the demo runs a
single env via the gymnasium backend.
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

# Currently forced to 1 — see the module docstring for the framework
# limitation. Leaving the constant in place so the multi-env enhancement
# only flips this knob.
NUM_ENVS = 1


def main() -> None:
    """Run the multi-env CartPole PPO demo."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    key = jax.random.key(SEED)
    k_agent, k_fit = jax.random.split(key)

    agent = build_agent(**AGENT_CFG, key=k_agent)
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

    print(f"Training PPO on CartPole-v1 (gymnasium backend, num_envs={NUM_ENVS})...")
    trainer.fit(k_fit)
    print(f"Done. Results saved to {RUN_DIR}/")


if __name__ == "__main__":
    main()
