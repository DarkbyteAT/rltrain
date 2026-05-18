r"""CLI for rltrain — thin Typer wrapper around the Trainer API.

Usage::

    python -m rltrain.cli train --agent examples/cartpole/ppo.json \
        --env examples/cartpole/env.json --dump results/
"""

from __future__ import annotations

import json
import logging
import sys
import time
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import jax
import typer

import rltrain.builders as mk
from rltrain.callbacks.checkpoint import CheckpointCallback
from rltrain.callbacks.csv_logger import CSVLoggerCallback
from rltrain.callbacks.plot import PlotCallback
from rltrain.trainer import Trainer


DT_SAVE = "%Y-%m-%d_%H-%M-%S"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

app = typer.Typer(add_completion=False)


class LogLevel(StrEnum):
    """Log level choices for the CLI ``--log-level`` option."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@app.command()
def train(
    agents: Annotated[
        list[Path],
        typer.Option(
            "--agent",
            help="Path(s) to agent JSON config files.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    env: Annotated[
        Path,
        typer.Option(
            help="Path to environment JSON config file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    dump: Annotated[Path, typer.Option(help="Output directory for results.")],
    num_steps: Annotated[int, typer.Option(help="Total training environment steps.")] = 100_000,
    checkpoint_steps: Annotated[int, typer.Option(help="Steps between checkpoints.")] = 2_500,
    save_all: Annotated[bool, typer.Option(help="Save model checkpoints at every interval.")] = False,
    seed: Annotated[int, typer.Option(help="RNG seed for reproducibility. -1 uses current time.")] = -1,
    log_level: Annotated[LogLevel, typer.Option(help="Logging level.")] = LogLevel.INFO,
) -> None:
    """Train one or more JAX RL agents on an environment."""
    if seed == -1:
        seed = int(time.time())

    level = getattr(logging, log_level.value, logging.INFO)
    logging.basicConfig(stream=sys.stdout, level=level, format=LOG_FORMAT)
    log = logging.getLogger("train")

    env_str = env.read_text()
    env_cfg = json.loads(env_str)
    rl_env = mk.env(**env_cfg)

    for agent_path in agents:
        agent_str = agent_path.read_text()
        agent_cfg = json.loads(agent_str)

        start_time = time.time()
        key = jax.random.PRNGKey(seed)
        k_build, k_fit = jax.random.split(key)

        rl_agent = mk.agent(key=k_build, **agent_cfg)
        agent_name = type(rl_agent).__name__

        run_dir = dump / agent_name / time.strftime(DT_SAVE, time.gmtime(start_time))
        cfg_path = run_dir / "config"
        cfg_path.mkdir(parents=True, exist_ok=True)
        (cfg_path / "seed.txt").write_text(f"{seed}")
        (cfg_path / "agent.json").write_text(agent_str)
        (cfg_path / "env.json").write_text(env_str)

        trainer = Trainer(
            rl_agent,
            rl_env,
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
        trainer.fit(k_fit)
        log.info("finished training %s on %s!", agent_name, env_cfg.get("id", "unknown"))

    log.info("all agents' training completed!")


def main() -> None:
    """Entry point for the ``rltrain`` console script."""
    app()


if __name__ == "__main__":
    main()
