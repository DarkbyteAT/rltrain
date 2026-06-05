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

## Your first agent

Before the JSON-config CLI route, here's the framework from a Python script. Six pieces stitched together — environment, actor backbone, action head, critic, agent, trainer — then `trainer.fit(key)` runs end-to-end. The full file is at [`examples/tutorial_ppo_cartpole.py`](https://github.com/DarkbyteAT/rltrain/blob/main/examples/tutorial_ppo_cartpole.py); the walk-through below narrates each step.

### 1. Hyperparameters

```python
import jax
import optax

from rltrain.agents.ppo import PPO
from rltrain.callbacks.checkpoint import CheckpointCallback
from rltrain.callbacks.csv_logger import CSVLoggerCallback
from rltrain.callbacks.plot import PlotCallback
from rltrain.env import GymnaxEnv
from rltrain.heads import DiscreteHead
from rltrain.networks import MLP
from rltrain.trainer import Trainer

SEED = 42
NUM_STEPS = 100_000
CHECKPOINT_STEPS = 5_000

OBS_DIM = 4         # CartPole observation
NUM_ACTIONS = 2     # left / right
HIDDEN = 64
```

### 2. Environment

`GymnaxEnv` wraps gymnax — pure JAX, jittable, scannable. The trainer detects this via `env.capabilities` and picks the fast `ScanLoop` strategy automatically.

```python
env = GymnaxEnv("CartPole-v1")
```

### 3. Networks

PPO needs three things: an actor backbone (observation → features), an action head (features → distribution), and a critic (observation → value). `MLP` is a thin wrapper around `eqx.nn.MLP` with orthogonal initialisation. `DiscreteHead` returns a `distreqx.Categorical` from a single linear projection.

```python
key = jax.random.key(SEED)
k_actor, k_head, k_critic, k_fit = jax.random.split(key, 4)

actor = MLP(in_size=OBS_DIM, out_size=HIDDEN, width_size=HIDDEN, depth=1, key=k_actor)
action_head = DiscreteHead(feature_dim=HIDDEN, action_dim=NUM_ACTIONS, key=k_head)
critic = MLP(in_size=OBS_DIM, out_size=1, width_size=HIDDEN, depth=1, key=k_critic)
```

### 4. Agent

`PPO` is an `eqx.Module` with no array leaves on `self` — it stores network architecture, the optimizer, and static hyperparameters. Mutable training state (params, opt_state, target_params) lives in a `TrainState` pytree that flows through `learn` as the scan carry.

```python
agent = PPO(
    actor=actor,
    action_head=action_head,
    critic=critic,
    optimizer=optax.adam(3e-4),
    gamma=0.99,
    tau=0.01,
    beta_critic=0.5,
    lambda_gae=0.95,
    eps_clip=0.2,
    num_epochs=4,
    minibatch_size=64,
)
```

### 5. Trainer

`Trainer` owns the loop. Pass agent + env + the three built-in callbacks (CSV metrics, return plots, model checkpoints) and `Trainer.fit(key)` does the rest.

```python
trainer = Trainer(
    agent,
    env,
    num_steps=NUM_STEPS,
    checkpoint_steps=CHECKPOINT_STEPS,
    callbacks=[
        CSVLoggerCallback(),
        PlotCallback(num_steps=NUM_STEPS),
        CheckpointCallback(),
    ],
    seed=SEED,
)
```

### 6. Train and evaluate

`fit` returns the final `TrainState`. Side effects (CSV writes, plots, checkpoints) fire from the callbacks at segment boundaries.

```python
final_state = trainer.fit(k_fit)

# Quick eval: one greedy rollout
eval_key = jax.random.key(SEED + 1)
es = env.reset(eval_key)
ep_return = 0.0
for _ in range(500):
    eval_key, k_act, k_step = jax.random.split(eval_key, 3)
    action = agent.act(final_state, es.obs, k_act)
    es = env.step(es, action, k_step)
    ep_return += float(es.reward)
    if bool(es.done):
        break
print(f"Eval episode return: {ep_return:.1f}")
```

Running `examples/tutorial_ppo_cartpole.py` from the repo root should print an eval return around ~450 within 100k steps (perfect play is 500).

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
