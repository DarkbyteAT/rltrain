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
python -m rltrain.cli \
    --agent examples/cartpole/ppo.json \
    --env examples/cartpole/env.json \
    --dump results/
```

JAX selects accelerators automatically (CPU by default; CUDA / TPU when available and configured via standard JAX env vars). Train multiple agents sequentially by repeating `--agent`:

```bash
python -m rltrain.cli \
    --agent examples/cartpole/ppo.json \
    --agent examples/cartpole/reinforce.json \
    --env examples/cartpole/env.json \
    --dump results/
```

## Understanding the output

Each training run produces a timestamped directory under `<dump>/<agent_name>/`:

```
results/PPO/2026-05-18_14-30-00/
    config/
        agent.json          # Copy of the agent config used
        env.json            # Copy of the environment config used
        seed.txt            # RNG seed for reproducibility
    models/
        model_FINAL.eqx     # Final TrainState pytree (eqx.tree_serialise_leaves)
        model_{step}.eqx    # Intermediate checkpoints (with save_all=True)
    metrics.csv             # Episode-level metrics (length, return, running_return)
    per_episode.svg         # Return vs episode plot
    per_sample.svg          # Return vs timestep plot
    videos/                 # Evaluation videos (when VideoRecorderCallback enabled)
```

`metrics.csv` has one row per episode: `episode, length, return, running_return`. The SVG plots visualise these.

## Programmatic usage

For more control, use the `Trainer` API directly:

```python
import jax
from rltrain.trainer import Trainer
from rltrain.callbacks.checkpoint import CheckpointCallback
from rltrain.callbacks.csv_logger import CSVLoggerCallback
from rltrain.callbacks.plot import PlotCallback

trainer = Trainer(
    agent,
    env,
    num_steps=100_000,
    checkpoint_steps=2_500,
    run_dir="results/ppo/run_1",
    callbacks=[CSVLoggerCallback(), PlotCallback(), CheckpointCallback(save_all=True)],
)
trainer.fit(jax.random.key(42))
```

The trainer auto-detects the loop strategy from `env.capabilities`: `ScanLoop` for pure-JAX envs (e.g. gymnax), `PythonLoop` for gymnasium-backed envs.

## CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--agent` | required | Path(s) to agent JSON config files (repeat for multiple) |
| `--env` | required | Path to environment JSON config file |
| `--dump` | required | Output directory for results |
| `--num-steps` | 100,000 | Total training environment steps |
| `--checkpoint-steps` | 2,500 | Steps between checkpoint hooks |
| `--seed` | current time | RNG seed for reproducibility |
| `--save-all` / `--no-save-all` | false | Save checkpoints at every interval |
| `--log-level` | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |

Run `python -m rltrain.cli --help` for the full auto-generated help.
