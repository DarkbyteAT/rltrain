"""Thin CLI wrapper around the Trainer API.

Usage:
    python run.py --agent examples/cartpole/ppo.json --env examples/cartpole/env.json --dump results/
    python run.py --agent examples/cartpole/ppo.json --agent examples/cartpole/reinforce.json \
        --env examples/cartpole/env.json --dump results/
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Annotated

import typer

import rltrain.utils.builders as mk
from rltrain.callbacks.checkpoint import CheckpointCallback
from rltrain.callbacks.csv_logger import CSVLoggerCallback
from rltrain.callbacks.plot import PlotCallback
from rltrain.env import MDP
from rltrain.trainer import Trainer
from rltrain.utils.device import resolve_device


DT_SAVE = "%Y-%m-%d_%H-%M-%S (%z)"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

app = typer.Typer(add_completion=False)


@app.command()
def train(
    agent: Annotated[list[Path], typer.Option(help="Path(s) to agent JSON config files.")],
    env: Annotated[Path, typer.Option(help="Path to environment JSON config file.")],
    dump: Annotated[Path, typer.Option(help="Output directory for results.")],
    num_steps: Annotated[int, typer.Option(help="Total training environment steps.")] = 100_000,
    checkpoint_steps: Annotated[int, typer.Option(help="Steps between saving metrics and plots.")] = 2_500,
    reward_run_rate: Annotated[float, typer.Option(help="EMA beta for running average return.")] = 0.1,
    img: Annotated[bool, typer.Option(help="Channel-first preprocessing for image observations.")] = False,
    device: Annotated[str, typer.Option(help="Device backend: cpu, cuda, mps, or auto.")] = "auto",
    workers: Annotated[int, typer.Option(help="PyTorch inter/intra-op thread count.")] = 12,
    log_freq: Annotated[int, typer.Option(help="Logging frequency (episodes).")] = 1,
    log_level: Annotated[str, typer.Option(help="Logging level (DEBUG, INFO, WARNING, ERROR).")] = "INFO",
    save_all: Annotated[bool, typer.Option(help="Save model checkpoints at every interval.")] = False,
    seed: Annotated[int, typer.Option(help="RNG seed for reproducibility.")] = -1,
) -> None:
    """Train one or more RL agents on an environment."""
    import seaborn as sns
    import torch as T

    if seed == -1:
        seed = int(time.time())

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    T.backends.cudnn.deterministic = True
    sns.set_style("darkgrid")
    sns.set_context("paper")

    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(stream=sys.stdout, level=level, format=LOG_FORMAT)
    log = logging.getLogger("Train")

    dev = resolve_device(device)
    log.info("device=%s", dev)

    if workers != 1:
        T.set_num_threads(workers)
        T.set_num_interop_threads(workers)

    for agent_path in agent:
        agent_str = agent_path.read_text()
        env_str = env.read_text()
        agent_cfg = json.loads(agent_str)
        env_cfg = json.loads(env_str)

        start_time = time.time()
        rl_agent = mk.agent(device=dev, **agent_cfg)
        mdp = MDP(mk.env(**env_cfg), reward_run_rate, log_freq, img)

        run_dir = dump / rl_agent.name / time.strftime(DT_SAVE, time.gmtime(start_time))
        cfg_path = run_dir / "config"
        cfg_path.mkdir(parents=True, exist_ok=True)
        (cfg_path / "seed.txt").write_text(f"{seed}")
        (cfg_path / "agent.json").write_text(agent_str)
        (cfg_path / "env.json").write_text(env_str)

        trainer = Trainer(
            rl_agent,
            mdp,
            num_steps=num_steps,
            checkpoint_steps=checkpoint_steps,
            run_dir=run_dir,
            callbacks=[
                CSVLoggerCallback(),
                PlotCallback(num_steps=num_steps),
                CheckpointCallback(save_all=save_all),
            ],
            seed=seed,
        )
        trainer.fit()
        log.info("finished training %s on %s!", rl_agent.name, env_cfg["id"])

    log.info("all agents' training completed!")


if __name__ == "__main__":
    app()
