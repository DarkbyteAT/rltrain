"""Train an RL agent with video recording.

Usage:
    python examples/video_demo.py
    python examples/video_demo.py --agent examples/cartpole/ppo.json --env examples/cartpole/env.json
    python examples/video_demo.py --agent examples/cartpole/ppo-sam.json --env examples/cartpole/env.json --steps 200000
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer

import rltrain.utils.builders as mk
from rltrain.callbacks.checkpoint import CheckpointCallback
from rltrain.callbacks.csv_logger import CSVLoggerCallback
from rltrain.callbacks.plot import PlotCallback
from rltrain.callbacks.video_recorder import VideoRecorderCallback
from rltrain.env import MDP
from rltrain.trainer import Trainer
from rltrain.utils.device import resolve_device


app = typer.Typer(add_completion=False)


@app.command()
def main(
    agent: Annotated[Path, typer.Option(help="Path to agent JSON config file.")] = Path(
        "examples/cartpole/ppo-sam.json"
    ),
    env: Annotated[Path, typer.Option(help="Path to environment JSON config file.")] = Path(
        "examples/cartpole/env.json"
    ),
    steps: Annotated[int, typer.Option(help="Total training environment steps.")] = 500_000,
    checkpoint_steps: Annotated[int, typer.Option(help="Steps between checkpoints.")] = 25_000,
    seed: Annotated[int, typer.Option(help="RNG seed for reproducibility.")] = 42,
    device: Annotated[str, typer.Option(help="Device backend: cpu, cuda, mps, or auto.")] = "auto",
    output: Annotated[Path, typer.Option(help="Output directory for results.")] = Path("results/video_demo"),
) -> None:
    """Train an RL agent with video recording at each checkpoint."""
    agent_str = agent.read_text()
    env_str = env.read_text()
    agent_cfg = json.loads(agent_str)
    env_cfg = json.loads(env_str)

    dev = resolve_device(device)
    run_dir = output / datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")

    rl_agent = mk.agent(device=dev, **agent_cfg)
    mdp = MDP(mk.env(**env_cfg), run_beta=0.05, log_freq=10, swap_channels=False)

    trainer = Trainer(
        rl_agent,
        mdp,
        num_steps=steps,
        checkpoint_steps=checkpoint_steps,
        run_dir=run_dir,
        callbacks=[
            CSVLoggerCallback(),
            PlotCallback(num_steps=steps),
            CheckpointCallback(),
            VideoRecorderCallback(
                env_fn=lambda: mk.eval_env(**env_cfg),
                num_episodes=1,
            ),
        ],
        seed=seed,
    )

    env_id = env_cfg.get("id", "unknown")
    print(f"Training {rl_agent.name} on {env_id} for {steps:,} steps...")
    print(f"Videos will be saved to {run_dir / 'videos'}/")
    trainer.fit()

    print(f"\nDone! Final running return: {mdp.run_reward:.1f}")
    print(f"Videos: {list((run_dir / 'videos').glob('*.mp4'))}")


if __name__ == "__main__":
    app()
