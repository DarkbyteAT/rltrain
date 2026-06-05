"""Train any rltrain agent with periodic video recording.

Usage:
    python examples/video_demo.py
    python examples/video_demo.py --agent examples/cartpole/ppo.json --env examples/cartpole/env.json
    python examples/video_demo.py --agent examples/acrobot/ppo.json --env examples/acrobot/env.json --steps 200000
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import gymnasium
import jax
import typer

from rltrain.builders import agent as build_agent
from rltrain.builders import env as build_env
from rltrain.callbacks.checkpoint import CheckpointCallback
from rltrain.callbacks.csv_logger import CSVLoggerCallback
from rltrain.callbacks.plot import PlotCallback
from rltrain.callbacks.video_recorder import VideoRecorderCallback
from rltrain.trainer import Trainer


app = typer.Typer(add_completion=False)


@app.command()
def main(
    agent: Annotated[
        Path,
        typer.Option(
            help="Path to agent JSON config file.", exists=True, file_okay=True, dir_okay=False, readable=True
        ),
    ] = Path("examples/cartpole/ppo.json"),
    env: Annotated[
        Path,
        typer.Option(
            help="Path to environment JSON config file.", exists=True, file_okay=True, dir_okay=False, readable=True
        ),
    ] = Path("examples/cartpole/env.json"),
    steps: Annotated[int, typer.Option(help="Total training environment steps.")] = 500_000,
    checkpoint_steps: Annotated[int, typer.Option(help="Steps between checkpoints.")] = 25_000,
    seed: Annotated[int, typer.Option(help="RNG seed for reproducibility.")] = 42,
    output: Annotated[Path, typer.Option(help="Output directory for results.")] = Path("results/video_demo"),
) -> None:
    """Train an RL agent with video recording at each checkpoint."""
    agent_cfg = json.loads(agent.read_text())
    env_cfg = json.loads(env.read_text())

    run_dir = output / datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    key = jax.random.key(seed)
    k_agent, k_fit = jax.random.split(key)

    rl_agent = build_agent(**agent_cfg, key=k_agent)
    rl_env = build_env(**env_cfg)

    trainer = Trainer(
        rl_agent,
        rl_env,
        num_steps=steps,
        checkpoint_steps=checkpoint_steps,
        run_dir=run_dir,
        callbacks=[
            CSVLoggerCallback(),
            PlotCallback(num_steps=steps),
            CheckpointCallback(),
            VideoRecorderCallback(
                env_fn=lambda: gymnasium.make(env_cfg["id"], render_mode="rgb_array"),
                num_episodes=1,
            ),
        ],
        seed=seed,
    )

    env_id = env_cfg.get("id", "unknown")
    print(f"Training {type(rl_agent).__name__} on {env_id} for {steps:,} steps...")
    print(f"Videos will be saved to {run_dir / 'videos'}/")
    trainer.fit(k_fit)
    print(f"Done. Results saved to {run_dir}/")


if __name__ == "__main__":
    app()
