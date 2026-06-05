# Configuration

RLTrain uses JSON config files with fully-qualified class names (FQNs) to specify every component at runtime. Algorithms, networks, optimisers, and environments can be swapped without changing any Python code.

## The FQN system

Every JSON object with an `fqn` field is resolved dynamically by `rltrain.builders.agent`. The FQN is a standard Python dotted import path:

```json
{"fqn": "rltrain.agents.PPO"}
```

The builder imports the module, retrieves the class, and constructs it with the remaining fields as kwargs. Sub-objects with their own `fqn` are recursively constructed and threaded into the parent. If a constructor accepts a `key=` parameter (detected via `inspect`), a PRNG sub-key is auto-spliced in.

## Agent config

```json
{
  "fqn": "rltrain.agents.PPO",
  "gamma": 0.99,
  "lambda_gae": 0.95,
  "eps_clip": 0.2,
  "num_epochs": 8,
  "minibatch_size": 128,
  "epoch_terminators": [
    {"fqn": "rltrain.agents.KLEarlyStop", "target_kl": 0.05, "rollback": true}
  ],
  "actor":        {"fqn": "rltrain.networks.MLP", "in_size": 4, "out_size": 64, "width_size": 256, "depth": 3},
  "action_head":  {"fqn": "rltrain.heads.DiscreteHead", "feature_dim": 64, "action_dim": 2},
  "critic":       {"fqn": "rltrain.networks.MLP", "in_size": 4, "out_size": 1,  "width_size": 256, "depth": 3},
  "optimizer":    {"fqn": "optax.adam", "learning_rate": 3e-4}
}
```

### Common fields

Which fields are valid depends on the agent class — see [Algorithms](algorithms.md) for the per-algorithm hyperparameter surface. Most on-policy agents accept:

| Field | Type | Description |
|---|---|---|
| `fqn` | string | Fully-qualified agent class name |
| `gamma` | float | Discount factor |
| `tau` | float | Entropy coefficient (PG family) or Polyak rate (DQN/SAC) |
| `actor` / `critic` | object | Network configs — usually `rltrain.networks.MLP` or a toblox FQN |
| `action_head` | object | Distribution head (`DiscreteHead`, `GaussianHead`, `SquashedGaussianHead`, `CategoricalAtomHead`, etc.) |
| `optimizer` | object | An `optax.GradientTransformation` constructor (e.g. `optax.adam`) |

SAC carries three separate optimisers (`actor_optimizer`, `critic_optimizer`, `alpha_optimizer`). C51 carries a `feature_net` plus a `CategoricalAtomHead` instead of `actor`/`critic`.

### Networks

The default MLP is `rltrain.networks.MLP` (orthogonal-init wrapper around `eqx.nn.MLP`). For SkipMLP / CNN / RFF and other architectures, use toblox FQNs — the builder resolves any importable class.

```json
{"fqn": "rltrain.networks.MLP", "in_size": 4, "out_size": 64, "width_size": 256, "depth": 3}
```

### Action heads

| Head | When |
|---|---|
| `rltrain.heads.DiscreteHead` | Categorical policies (discrete action spaces) |
| `rltrain.heads.GaussianHead` | Diagonal Gaussian (continuous, unbounded) |
| `rltrain.heads.SquashedGaussianHead` | SAC continuous (tanh-squashed Gaussian) |
| `rltrain.heads.GammaHead` / `BetaHead` | Bounded / non-negative continuous |
| `rltrain.heads.CategoricalAtomHead` | C51 (distributional DQN) |

### Optimiser

The builder constructs any [optax](https://optax.readthedocs.io/) optimiser via FQN:

```json
{"fqn": "optax.adam", "learning_rate": 3e-4}
```

To compose gradient transforms (e.g. samgria's SAM, gradient clipping), wrap with `optax.chain`:

```json
{
  "fqn": "optax.chain",
  "transformations": [
    {"fqn": "optax.clip_by_global_norm", "max_norm": 0.5},
    {"fqn": "optax.adam", "learning_rate": 3e-4}
  ]
}
```

## Environment config

```json
{"backend": "gymnax", "id": "CartPole-v1"}
```

The `backend` key dispatches between two env classes:

- `gymnax` — pure-step, jittable, scannable. The trainer auto-selects `ScanLoop` for in-XLA rollouts.
- `gymnasium` — Python-loop, broad env coverage. Trainer uses `PythonLoop`.

Additional fields are passed to the env constructor (e.g. `params` for gymnax, `wrappers` for gymnasium).

## PPO epoch terminators

PPO accepts a list of composable `EpochTerminator` instances. Each is a callable returning a bool indicating whether to stop the epoch loop early.

```json
{
  "epoch_terminators": [
    {"fqn": "rltrain.agents.KLEarlyStop", "target_kl": 0.05, "rollback": true}
  ]
}
```

`KLEarlyStop` halts mini-batch epochs when the approximate KL between current and pre-epoch policy exceeds `target_kl`. With `rollback: true`, it restores the pre-epoch parameters.

## Out-of-domain features

- **Gradient transforms** (SAM/ASAM/LAMP) live in [samgria](https://github.com/DarkbyteAT/samgria) — JAX-native optax-compatible primitives. Plug into the `optimizer` field via `optax.chain`.
- **Network architectures** (SkipMLP, CNN, RFF) live in [toblox](https://github.com/DarkbyteAT/toblox). Reference via FQN in `actor`/`critic`.
- **Experiment tracking** (W&B, TensorBoard, DuckDB) lives in [xptrack](https://github.com/DarkbyteAT/xptrack). Add as a callback alongside `CSVLoggerCallback`.
