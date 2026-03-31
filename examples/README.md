# Examples

Experiment configurations from the original dissertation (COMP3200, University of Southampton 2022). Each directory contains an `env.json` and one or more agent configs.

## Environments

| Directory | Environment | Type |
|-----------|------------|------|
| `cartpole/` | CartPole-v1 | Classic control |
| `acrobot/` | Acrobot-v1 | Classic control |
| `breakout/` | Breakout-MinAtar-v0 | MinAtar (requires `minatar`) |
| `invaders/` | SpaceInvaders-MinAtar-v0 | MinAtar (requires `minatar`) |
| `catcher/` | Catcher-PLE-v0 | PLE (requires `ple`) |
| `pixelcopter/` | Pixelcopter-PLE-v0 | PLE (requires `ple`) |
| `rff/` | Random Fourier Features experiment | PLE |

## Running an Experiment

```bash
# Train PPO on CartPole
python run.py --agents examples/cartpole/ppo.json --env examples/cartpole/env.json --dump results/

# Train REINFORCE with SAM on Acrobot
python run.py --agents examples/acrobot/reinforce-sam.json --env examples/acrobot/env.json --dump results/

# Train multiple agents sequentially
python run.py --agents examples/cartpole/ppo.json examples/cartpole/a2c.json --env examples/cartpole/env.json --dump results/
```

## Using the Trainer API

```python
import json
import torch as T
import rltrain.utils.builders as mk
from rltrain.env import MDP
from rltrain.trainer import Trainer

agent_cfg = json.loads(open("examples/cartpole/ppo.json").read())
env_cfg = json.loads(open("examples/cartpole/env.json").read())

agent = mk.agent(device=T.device("cpu"), **agent_cfg)
env = MDP(mk.env(**env_cfg), run_beta=0.1, log_freq=10, swap_channels=False)

trainer = Trainer(agent, env, num_steps=100_000, checkpoint_steps=2500,
                  run_dir="results/", seed=42)
trainer.fit()
```

## Agent Variants

Each environment has configs for multiple algorithms:
- `reinforce.json` — REINFORCE with learned baseline
- `a2c.json` — Advantage Actor-Critic with GAE
- `ppo.json` — Proximal Policy Optimisation
- `*-sam.json` — With SAM (Sharpness-Aware Minimisation)
- `*-lamp.json` — With LAMP (Local-Averaging over Multiple Perturbations)

## Dissertation Artifacts

- `full_results.csv` — Aggregated experiment results from the dissertation
- `metrics.csv` — Sample training metrics
- `plot.py` — Custom plotting script used for dissertation figures
- `job.bat` — Windows batch script for running experiments on IRIDIS HPC
