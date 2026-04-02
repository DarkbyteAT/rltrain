import json
import logging
import os
import sys
import time
from pathlib import Path

import seaborn as sns
import torch as T
from tap import Tap

import rltrain.utils.builders as mk
from rltrain.callbacks.checkpoint import CheckpointCallback
from rltrain.callbacks.csv_logger import CSVLoggerCallback
from rltrain.callbacks.plot import PlotCallback
from rltrain.env import MDP
from rltrain.trainer import Trainer
from rltrain.utils.device import resolve_device


DT_SAVE = "%Y-%m-%d_%H-%M-%S (%z)"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"


class Parser(Tap):
    agents: list[Path]
    env: Path
    dump: Path

    num_steps: int = 100000
    checkpoint_steps: int = 2500
    reward_run_rate: float = 0.1
    img: bool = False
    device: str = "auto"

    workers: int = 12
    log_freq: int = 1
    log_level = logging.INFO
    save_all: bool = False
    seed: int = int(time.time())


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
T.backends.cudnn.deterministic = True
sns.set_style("darkgrid")
sns.set_context("paper")

args = Parser(fromfile_prefix_chars="@").parse_args()
level = args.log_level if isinstance(args.log_level, int) else getattr(logging, args.log_level)
logging.basicConfig(stream=sys.stdout, level=level, format=LOG_FORMAT)
log = logging.getLogger("Train")

device = resolve_device(args.device)
log.info("device=%s", device)

if args.workers != 1:
    T.set_num_threads(args.workers)
    T.set_num_interop_threads(args.workers)


def train(args: Parser) -> None:
    for agent_path in args.agents:
        agent_str = agent_path.read_text()
        env_str = args.env.read_text()
        agent_cfg = json.loads(agent_str)
        env_cfg = json.loads(env_str)

        start_time = time.time()
        agent = mk.agent(device=device, **agent_cfg)
        env = MDP(mk.env(**env_cfg), args.reward_run_rate, args.log_freq, args.img)

        run_dir = args.dump / agent.name / time.strftime(DT_SAVE, time.gmtime(start_time))
        cfg_path = run_dir / "config"
        cfg_path.mkdir(parents=True, exist_ok=True)
        (cfg_path / "seed.txt").write_text(f"{args.seed}")
        (cfg_path / "agent.json").write_text(agent_str)
        (cfg_path / "env.json").write_text(env_str)

        trainer = Trainer(
            agent,
            env,
            num_steps=args.num_steps,
            checkpoint_steps=args.checkpoint_steps,
            run_dir=run_dir,
            callbacks=[
                CSVLoggerCallback(),
                PlotCallback(num_steps=args.num_steps),
                CheckpointCallback(save_all=args.save_all),
            ],
            seed=args.seed,
        )
        trainer.fit()
        log.info("finished training %s on %s!", agent.name, env_cfg["id"])

    log.info("all agents' training completed!")


if __name__ == "__main__":
    train(args)
