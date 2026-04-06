# Configuration

RLTrain uses JSON configuration files with fully-qualified class names (FQNs) to specify every component at runtime. This means you can swap algorithms, network architectures, optimisers, and environment wrappers without changing any Python code.

## The FQN system

Every JSON object with an `fqn` field is resolved dynamically by the builder. The FQN is a standard Python dotted import path:

```json
{"fqn": "rltrain.agents.actor_critic.PPO"}
```

At runtime, the builder imports `rltrain.agents.actor_critic` and retrieves the `PPO` class. All remaining fields in the JSON object are passed as constructor arguments. This works for any importable class, including your own custom agents, networks, and optimisers.

## Agent config anatomy

An agent config file has three sections: algorithm hyperparameters, model architecture, and optimiser settings.

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

### Top-level fields

These are passed directly to the agent constructor. Which fields are available depends on the agent class -- see the [Algorithms](algorithms.md) page for per-algorithm hyperparameters.

Common fields shared across most agents:

| Field | Type | Description |
|-------|------|-------------|
| `fqn` | string | Fully-qualified class name of the agent |
| `gamma` | float | Discount factor for future rewards |
| `tau` | float | Entropy regularisation coefficient (policy gradient) or soft update rate (DQN) |
| `normalise` | bool | Standardise returns/advantages to zero mean and unit variance |
| `continuous` | bool | Whether the action space is continuous (Gaussian) or discrete (categorical) |

### Model section

The `model` section maps network roles to lists of module specs. Each list entry becomes a layer in an `nn.Sequential`, so modules compose naturally.

**Actor-critic agents** expect `actor` and `critic` keys. Optionally, an `embedding` key defines shared feature layers (useful for visual observations):

```json
{
    "model": {
        "embedding": [
            {"fqn": "rltrain.nn.cnn", "channels": [4, 16, 32], "kernels": [2, 2], "strides": [2, 1]}
        ],
        "actor": [
            {"fqn": "rltrain.nn.SkipMLP", "inputs": 512, "hiddens": [256, 256, 256, 256], "outputs": 3}
        ],
        "critic": [
            {"fqn": "rltrain.nn.SkipMLP", "inputs": 512, "hiddens": [256, 256, 256, 256], "outputs": 1}
        ]
    }
}
```

**DQN agents** expect a single `q` key (and optionally `embedding`).

### Optimiser section

The `opt` section maps the same network roles to optimiser specs:

```json
{
    "opt": {
        "actor": {"fqn": "torch.optim.Adam", "lr": 3e-4},
        "critic": {"fqn": "torch.optim.Adam", "lr": 3e-4}
    }
}
```

Any `torch.optim.Optimizer` subclass works. Pass constructor arguments (learning rate, weight decay, etc.) as additional fields.

## Environment config

Environment configs specify a Gymnasium environment ID and an optional list of wrappers:

```json
{
    "id": "CartPole-v1",
    "wrappers": [
        {"fqn": "gymnasium.wrappers.NormalizeObservation"}
    ]
}
```

Wrappers are applied in order. Any `gymnasium.Wrapper` subclass works, including custom wrappers importable from your code.

## Network modules

RLTrain ships with four network modules, all using orthogonal weight initialisation and SiLU activation by default.

### MLP

```json
{"fqn": "rltrain.nn.mlp", "inputs": 4, "hiddens": [128, 128], "outputs": 2}
```

Standard multi-layer perceptron. Best for low-dimensional dense observations (CartPole, Acrobot).

### SkipMLP (D2RL)

```json
{"fqn": "rltrain.nn.SkipMLP", "inputs": 4, "hiddens": [256, 256, 256, 256], "outputs": 2}
```

D2RL-style MLP with skip connections from the input to every hidden layer. Improves gradient flow in deeper networks[^d2rl].

### CNN

```json
{"fqn": "rltrain.nn.cnn", "channels": [4, 16, 32], "kernels": [2, 2], "strides": [2, 1]}
```

Convolutional network with a flatten layer at the output. Use as an embedding network for image observations.

### RFF (Random Fourier Features)

```json
{"fqn": "rltrain.nn.RFF", "inputs": 4, "features": 256}
```

Projects inputs through a fixed random matrix with sinusoidal activations, approximating a kernel feature map. Useful for spectral encoding of low-dimensional inputs[^rff].

## Gradient transforms

Add composable gradient transforms to any agent via the `grad_transforms` key. Transforms run between `loss.backward()` and `optimizer.step()` inside `Agent.learn()`, and are specified as a list of FQN objects:

```json
{
    "grad_transforms": [
        {"fqn": "samgria.SAM", "rho": 0.01},
        {"fqn": "samgria.LAMPRollback", "eps": 5e-3, "rollback_len": 10}
    ]
}
```

| Transform | Class | Parameters | Description |
|-----------|-------|------------|-------------|
| SAM | `samgria.SAM` | `rho` (perturbation radius) | Perturbs parameters in the gradient direction, recomputes loss at the perturbed point, then descends using the gradient computed there |
| ASAM | `samgria.ASAM` | `rho` (perturbation radius) | Like SAM but perturbation is scaled by parameter magnitude for scale-invariant sharpness |
| LAMP | `samgria.LAMPRollback` | `eps` (noise scale), `rollback_len` (rollback interval) | Injects parameter noise after each step and periodically rolls back to a moving average |

Transforms compose -- list them in order. SAM/ASAM use the `apply()` hook (pre-descent) and LAMP uses the `post_step()` hook (post-descent), so they naturally complement each other[^sam].

[^d2rl]: Sinha, S. et al. (2020). D2RL: Deep Dense Architectures in Reinforcement Learning. *arXiv:2010.09163*.
[^rff]: Rahimi, A. & Recht, B. (2007). Random Features for Large-Scale Kernel Machines. *NeurIPS*, 20.
[^sam]: Foret, P. et al. (2021). Sharpness-Aware Minimization for Efficiently Improving Generalization. *ICLR*.
