# Env

Environment interface layer that wraps gymnasium into rltrain's training loop. Provides the `MDP` wrapper and the `Trajectory` data container.

## MDP

`MDP` (defined in `mdp.py`) wraps a `gymnasium.vector.SyncVectorEnv` and provides:

- **Automatic reset** -- the environment resets itself when an episode ends, so agents never deal with reset logic.
- **Unified done signal** -- gymnasium's separate `terminated` and `truncated` flags are combined into a single `done` boolean at the MDP level.
- **Episode statistics** -- tracks `length_history`, `return_history`, `run_history` (EMA of returns), `episode_count`, `total_steps`, and `episode_steps`.
- **Observation preprocessing** -- optional channel-first transposition for image environments via the `swap_channels` flag.

### `MDP.step(policy)`

The main interaction method. Takes a policy callable `(ndarray) -> ndarray`, executes one environment step, updates internal statistics, handles episode boundaries, and returns a `Trajectory`.

Agents are callable (via `Agent.__call__`), so the typical usage is `env.step(agent)`.

### Construction

`MDP` is not constructed directly by user code. The builder function `rltrain.utils.builders.env()` creates the `SyncVectorEnv` from a JSON config and the `Trainer` wraps it in an `MDP`.

## Trajectory

`Trajectory` (defined in `trajectory.py`) is a frozen, generic dataclass representing a single transition:

```
Trajectory[U](state, action, reward, next_state, done)
```

It is generic over `U` (typically `np.ndarray` when returned from `MDP.step()`, or `torch.Tensor` after batching). It supports iteration via `__iter__`, which yields the five fields in order -- this is what enables `tuple(T.from_numpy(...) for x in zip(*memory))` unpacking in agent `load()` methods.

## Known Limitation

The current implementation wraps a single environment in `SyncVectorEnv`, so there is no true parallelism. This is tracked as technical debt.
