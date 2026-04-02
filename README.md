# RLTrain

A modular PyTorch framework for deep reinforcement learning research. JSON-driven configuration, composable neural network architectures, and robust optimisation techniques — designed for comparing RL algorithms across environments with minimal boilerplate.

## Overview

RLTrain separates *what* you train (algorithm + architecture) from *where* you train it (environment) using JSON configuration files with fully-qualified class names. Every component — agent, network, optimiser, environment wrapper — is resolved dynamically at runtime, so new algorithms and architectures slot in without touching the training loop.

```mermaid
graph LR
    subgraph Configuration
        AJ["agent.json"] --> B["Builder"]
        EJ["env.json"] --> B
    end

    subgraph Framework
        B --> TR["Trainer"]
        TR --> A["Agent"]
        TR --> E["MDP Environment"]
        TR -->|"hooks"| CB["Callbacks"]
        A -->|"act(states)"| P["Policy π(a|s)"]
        P -->|"action"| E
        E -->|"Trajectory (s,a,r,s',done)"| A
        A -->|"loss → learn → descend"| A
    end

    subgraph "Robust Optimisation"
        A -.->|"optional"| SAM["SAM / LAMP"]
    end
```

## Algorithms

| Algorithm | Class | Method | Key Idea |
|-----------|-------|--------|----------|
| Vanilla Policy Gradient | `VanillaPG` | Policy gradient | REINFORCE without baseline, entropy regularisation |
| REINFORCE | `REINFORCE` | Policy gradient | Learned value baseline reduces variance |
| Vanilla Actor-Critic | `VanillaAC` | Actor-critic | TD error advantage, optional shared feature layers |
| Advantage Actor-Critic | `AdvantageAC` | Actor-critic | GAE (Generalised Advantage Estimation), horizon-based collection |
| PPO | `PPO` | Actor-critic | Clipped surrogate objective, mini-batch epochs, KL early stopping |
| DQN | `VanillaDQN` | Q-learning | Replay buffer, target network with soft updates, epsilon-greedy decay |

All policy gradient and actor-critic agents inherit along a clean chain:

```
Agent (ABC)
├── VanillaPG
│   └── REINFORCE
│       └── VanillaAC
│           └── AdvantageAC
│               └── PPO
└── VanillaDQN
```

Each level adds one concept — baselines, TD bootstrapping, GAE, clipping — making the hierarchy a readable tutorial in itself.

## Neural Network Modules

| Module | Description | Use Case |
|--------|-------------|----------|
| `mlp` | Multi-layer perceptron with orthogonal init | Dense observations (CartPole, Acrobot) |
| `cnn` | Convolutional network with flatten output | Image observations (MinAtar, PLE) |
| `SkipMLP` | D2RL-style MLP with skip connections from input to every hidden layer | Improved gradient flow for deeper networks |
| `RFF` | Random Fourier Features projection layer | Spectral feature encoding for low-dimensional inputs |

## Quick Start

### Installation

```bash
git clone https://github.com/DarkbyteAT/rltrain.git
cd rltrain

# Install with uv (recommended)
uv sync --group dev

# Or with pip
pip install -e ".[dev]"
```


### Training an Agent

**CLI** (thin wrapper around the Trainer API):

```bash
# Train PPO on CartPole (auto-detects best device: CUDA → MPS → CPU)
python run.py --agents examples/cartpole/ppo.json --env examples/cartpole/env.json --dump results/

# Explicitly select a device backend
python run.py --agents examples/cartpole/ppo.json --env examples/cartpole/env.json --dump results/ --device mps
python run.py --agents examples/cartpole/ppo.json --env examples/cartpole/env.json --dump results/ --device cuda

# Train multiple agents sequentially
python run.py --agents examples/cartpole/ppo.json examples/cartpole/reinforce.json --env examples/cartpole/env.json --dump results/
```

**Trainer API** (programmatic use):

```python
from rltrain.trainer import Trainer
from rltrain.callbacks.checkpoint import CheckpointCallback
from rltrain.callbacks.csv_logger import CSVLoggerCallback
from rltrain.callbacks.plot import PlotCallback
from rltrain.utils.device import resolve_device

device = resolve_device("auto")  # CUDA → MPS → CPU

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

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--agent` | required | Path(s) to agent JSON config files |
| `--env` | required | Path to environment JSON config file |
| `--dump` | required | Output directory for results |
| `--num_steps` | 100,000 | Total training environment steps |
| `--checkpoint_steps` | 2,500 | Steps between saving metrics and plots |
| `--reward_run_rate` | 0.1 | EMA beta for running average return |
| `--device` | auto | Device backend: `cpu`, `cuda`, `mps`, or `auto` (CUDA → MPS → CPU) |
| `--workers` | 12 | PyTorch inter/intra-op thread count |
| `--seed` | current time | RNG seed for reproducibility |
| `--img` | false | Channel-first preprocessing for image observations |
| `--save_all` | false | Save model checkpoints at every interval (not just final) |

## Configuration

### Agent Config (`agent.json`)

Agents are specified as JSON objects. The `fqn` field resolves to a Python class at runtime, so any class on the import path works — including your own.

```json
{
    "fqn": "rltrain.agents.actor_critic.PPO",
    "gamma": 0.995,
    "tau": 0.01,
    "beta_critic": 0.5,
    "normalise": true,
    "continuous": false,
    "horizon": 256,
    "lambda_gae": 0.95,
    "num_epochs": 8,
    "batch_size": 128,
    "early_stop": 0.05,
    "eps_clip": 0.2,
    "model": {
        "actor": [{"fqn": "rltrain.nn.SkipMLP", "inputs": 4, "hiddens": [256, 256, 256, 256], "outputs": 2}],
        "critic": [{"fqn": "rltrain.nn.SkipMLP", "inputs": 4, "hiddens": [256, 256, 256, 256], "outputs": 1}]
    },
    "opt": {
        "actor": {"fqn": "torch.optim.Adam", "lr": 3e-4},
        "critic": {"fqn": "torch.optim.Adam", "lr": 3e-4}
    }
}
```

Networks are composed sequentially — each entry in the model list becomes a layer in an `nn.Sequential`. For visual environments, use an embedding CNN followed by an MLP:

```json
{
    "model": {
        "embedding": [{"fqn": "rltrain.nn.cnn", "channels": [4, 16, 32], "kernels": [2, 2], "strides": [2, 1]}],
        "actor": [{"fqn": "rltrain.nn.SkipMLP", "inputs": 512, "hiddens": [256, 256, 256, 256], "outputs": 3}],
        "critic": [{"fqn": "rltrain.nn.SkipMLP", "inputs": 512, "hiddens": [256, 256, 256, 256], "outputs": 1}]
    }
}
```

### Environment Config (`env.json`)

```json
{
    "id": "CartPole-v1",
    "wrappers": [
        {"fqn": "gymnasium.wrappers.NormalizeObservation"}
    ]
}
```

Wrappers are applied in order. Any `gymnasium.Wrapper` subclass works, including custom wrappers.

### Robust Optimisation

Add SAM (Sharpness-Aware Minimization) or LAMP (Local Averaging over Multiple Perturbations) to any agent:

```json
{
    "robust": true,
    "rho_sam": 0.01,
    "rollback_len": 0
}
```

Set `rollback_len > 0` to use LAMP instead of SAM. LAMP adds parameter noise and periodically rolls back to a moving average, encouraging exploration of flatter loss regions.

## Callbacks

The training loop is extensible via the `Callback` protocol. Built-in callbacks handle checkpointing, CSV metrics, and SVG plots. Write a custom callback by implementing any subset of the hook methods:

```python
from pathlib import Path
from rltrain.agents.agent import Agent
from rltrain.callbacks import Callback
from rltrain.env import MDP


class WandbCallback:
    """Example custom callback that logs to Weights & Biases."""

    def on_train_start(self, agent: Agent, env: MDP, run_dir: Path) -> None:
        import wandb
        wandb.init(project="rltrain", name=agent.name)

    def on_episode_end(self, agent: Agent, env: MDP, episode: int) -> None:
        import wandb
        wandb.log({"return": env.return_history[-1], "episode": episode})

    def on_train_end(self, agent: Agent, env: MDP, run_dir: Path) -> None:
        import wandb
        wandb.finish()
```

Pass custom callbacks to the `Trainer`:

```python
trainer = Trainer(
    agent, env,
    num_steps=100_000,
    checkpoint_steps=2500,
    run_dir=run_dir,
    callbacks=[CSVLoggerCallback(), CheckpointCallback(), WandbCallback()],
    seed=42,
)
```

### Hook Methods

| Method | Called | Arguments |
|--------|--------|-----------|
| `on_train_start` | Once, before the loop begins | agent, env, run_dir |
| `on_step` | After every `agent.step()` call | agent, env, step |
| `on_episode_end` | When an episode completes | agent, env, episode |
| `on_checkpoint` | At checkpoint intervals | agent, env, run_dir |
| `on_train_end` | Once, after the loop exits | agent, env, run_dir |

## Output

Training produces the following in `<dump>/<agent_name>/<timestamp>/`:

```
config/
    agent.json          # Copy of agent config used
    env.json            # Copy of environment config used
    seed.txt            # RNG seed for reproducibility
models/
    model_FINAL.pt      # Final model state dict
    model_2500.pt       # Intermediate checkpoints (if --save_all)
metrics.csv             # Episode-level metrics (length, return, running return)
per_episode.svg         # Return vs episode plot
per_sample.svg          # Return vs timestep plot
```

## Architecture

```
rltrain/
├── agents/
│   ├── agent.py                # Abstract base — setup, act, step, load, loss, learn, descend
│   ├── policy_gradient/
│   │   ├── vanilla.py          # VanillaPG — entropy-regularised policy gradient
│   │   └── baseline.py         # REINFORCE — adds learned value baseline
│   ├── actor_critic/
│   │   ├── vanilla.py          # VanillaAC — TD error advantage, optional shared features
│   │   ├── a2c.py              # AdvantageAC — GAE, horizon-based collection
│   │   └── ppo.py              # PPO — clipped surrogate, mini-batch, KL early stop
│   └── q_learning/
│       └── vanilla.py          # VanillaDQN — replay buffer, target network, epsilon decay
├── callbacks/
│   ├── __init__.py             # Callback protocol (runtime_checkable)
│   ├── checkpoint.py           # CheckpointCallback — model state_dict saves
│   ├── csv_logger.py           # CSVLoggerCallback — episode metrics to CSV
│   └── plot.py                 # PlotCallback — per-episode and per-sample SVG plots
├── env/
│   ├── mdp.py                  # MDP wrapper — auto-reset, metric tracking, preprocessing
│   └── trajectory.py           # Trajectory dataclass — (s, a, r, s', done)
├── nn/
│   ├── mlp.py                  # MLP with orthogonal init
│   ├── cnn.py                  # CNN with flatten output
│   ├── d2rl.py                 # SkipMLP — D2RL skip connections
│   └── rff.py                  # Random Fourier Features layer
├── trainer.py                  # Trainer — training loop + callback orchestration
└── utils/
    ├── builders/               # Factory functions — dynamic FQN class loading
    ├── discount.py             # Discounted return / GAE computation
    ├── center.py               # Standardisation (zero mean, unit variance)
    ├── grad.py                 # Gradient vector get/set utilities
    └── lerp.py                 # Linear interpolation (for target network soft updates)
```

## Experiments

The repository includes pre-configured experiments from the original dissertation research, comparing REINFORCE, A2C, and PPO (each with baseline, SAM, and LAMP variants) across six environments:

| Environment | Type | Observation | Actions |
|-------------|------|-------------|---------|
| CartPole-v1 | Classic control | 4D vector | 2 |
| Acrobot-v1 | Classic control | 6D vector | 3 |
| Catcher-PLE-v0 | PLE game | 4D vector | 3 |
| Pixelcopter-PLE-v0 | PLE game | 7D vector | 2 |
| Breakout-MinAtar-v1 | MinAtar arcade | 10x10x4 image | 3 |
| SpaceInvaders-MinAtar-v1 | MinAtar arcade | 10x10x6 image | 4 |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code conventions, architecture rules, and key patterns.

## References

See [references/references.bib](references/references.bib) for the papers that inform this framework.

## Origin

RLTrain was built for COMP3200 (Individual Project) at the University of Southampton in 2022, investigating the effect of robust optimisation techniques (SAM and LAMP) on deep reinforcement learning across classic control and visual environments.

## License

MIT
