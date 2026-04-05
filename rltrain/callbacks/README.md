# Callbacks

Extensible hook system for the training loop. The `Trainer.fit()` method invokes callbacks at defined points during training, allowing observation and side-effects without modifying the core loop.

## Callback Protocol

`Callback` is a `@runtime_checkable` Protocol defined in `__init__.py`. The Trainer calls all five hooks unconditionally, so implementations must define all five methods. Use `...` as the body for hooks you don't need. No inheritance required — any class with the right method signatures satisfies the protocol via structural subtyping.

| Hook | When | Signature |
|------|------|-----------|
| `on_train_start` | Once before the loop | `(agent, env, run_dir)` |
| `on_step` | After every `agent.step()` | `(agent, env, step)` |
| `on_episode_end` | When an episode completes | `(agent, env, episode)` |
| `on_checkpoint` | At checkpoint intervals | `(agent, env, run_dir)` |
| `on_train_end` | Once after the loop exits | `(agent, env, run_dir)` |

## Built-in Callbacks

| Callback | File | Purpose |
|----------|------|---------|
| `CSVLoggerCallback` | `csv_logger.py` | Writes episode metrics (length, return, running return) to `metrics.csv` at each checkpoint |
| `PlotCallback` | `plot.py` | Renders per-episode and per-sample return SVG plots at each checkpoint |
| `CheckpointCallback` | `checkpoint.py` | Saves model `state_dict` at checkpoints and/or train end |
| `VideoRecorderCallback` | `video_recorder.py` | Records evaluation videos as MP4 via a separate rendering environment |

If no callbacks are passed to the `Trainer`, it defaults to `CSVLoggerCallback`, `PlotCallback`, and `CheckpointCallback`.

## How to Write a Custom Callback

1. Create a class with all five hook methods matching the signatures above. Use `...` for hooks you don't need.
2. No base class or decorator is needed -- structural subtyping handles protocol conformance.
3. Pass an instance to the `Trainer`'s `callbacks` list.

```python
class MyCallback:
    def on_episode_end(self, agent, env, episode):
        print(f"Episode {episode}: return={env.return_history[-1]:.2f}")
```

For experiment tracking backends, prefer using `TrackingCallback` from `rltrain/tracking/` rather than implementing raw callback hooks -- it provides a higher-level `MetricsLogger` interface.

## Conventions

- Callbacks must not modify agent or environment state. They are observers.
- Use `TYPE_CHECKING` guards for `Agent` and `MDP` type hints to avoid circular imports.
- Callbacks that need setup state (file paths, connections) should initialise in `on_train_start` and clean up in `on_train_end`.
