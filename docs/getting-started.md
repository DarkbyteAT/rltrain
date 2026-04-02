# Getting Started

## Installation

=== "uv (recommended)"

    ```bash
    git clone https://github.com/DarkbyteAT/rltrain.git
    cd rltrain
    uv sync
    ```

=== "pip"

    ```bash
    git clone https://github.com/DarkbyteAT/rltrain.git
    cd rltrain
    pip install -e .
    ```

For video recording support, install the `video` extra:

```bash
pip install -e ".[video]"
```

## First training run

RLTrain separates *what* you train (agent + architecture) from *where* you train it (environment) using two JSON config files.

The repository ships with example configs under `examples/`. Train PPO on CartPole:

```bash
python run.py \
    --agents examples/cartpole/ppo.json \
    --env examples/cartpole/env.json \
    --dump results/
```

RLTrain auto-detects the best available device (CUDA, then MPS, then CPU). Override with `--device`:

```bash
python run.py \
    --agents examples/cartpole/ppo.json \
    --env examples/cartpole/env.json \
    --dump results/ \
    --device mps
```

Train multiple agents sequentially by passing several config files:

```bash
python run.py \
    --agents examples/cartpole/ppo.json examples/cartpole/reinforce.json \
    --env examples/cartpole/env.json \
    --dump results/
```

## Understanding the output

Each training run produces a timestamped directory under `<dump>/<agent_name>/`:

```
results/PPO/2026-04-02_14-30-00/
    config/
        agent.json          # Copy of the agent config used
        env.json            # Copy of the environment config used
        seed.txt            # RNG seed for reproducibility
    models/
        model_FINAL.pt      # Final model state dict
        model_2500.pt       # Intermediate checkpoints (if --save_all)
    metrics.csv             # Episode-level metrics (length, return, running return)
    per_episode.svg         # Return vs episode plot
    per_sample.svg          # Return vs timestep plot
    videos/                 # Evaluation videos (if VideoRecorderCallback enabled)
```

`metrics.csv` contains one row per episode with columns for episode length, raw return, and exponentially-smoothed running return. The SVG plots visualise these metrics.

## Programmatic usage

For more control, use the `Trainer` API directly:

```python
from pathlib import Path

from rltrain.trainer import Trainer
from rltrain.callbacks.checkpoint import CheckpointCallback
from rltrain.callbacks.csv_logger import CSVLoggerCallback
from rltrain.callbacks.plot import PlotCallback
from rltrain.utils.device import resolve_device

device = resolve_device("auto")  # CUDA -> MPS -> CPU

trainer = Trainer(
    agent,
    env,
    num_steps=100_000,
    checkpoint_steps=2500,
    run_dir=Path("results/ppo/run_1"),
    callbacks=[
        CSVLoggerCallback(),
        PlotCallback(num_steps=100_000),
        CheckpointCallback(save_all=True),
    ],
    seed=42,
)
trainer.fit()
```

## CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--agents` | required | Path(s) to agent JSON config files |
| `--env` | required | Path to environment JSON config file |
| `--dump` | required | Output directory for results |
| `--num_steps` | 100,000 | Total training environment steps |
| `--checkpoint_steps` | 2,500 | Steps between saving metrics and plots |
| `--reward_run_rate` | 0.1 | EMA beta for running average return |
| `--device` | auto | Device backend: `cpu`, `cuda`, `mps`, or `auto` |
| `--workers` | 12 | PyTorch inter/intra-op thread count |
| `--seed` | current time | RNG seed for reproducibility |
| `--img` | false | Channel-first preprocessing for image observations |
| `--save_all` | false | Save model checkpoints at every interval |
