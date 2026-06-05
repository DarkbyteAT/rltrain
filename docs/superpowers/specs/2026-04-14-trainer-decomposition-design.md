# Trainer Decomposition Design

## Problem

The current `spike/trainer.py` is 502 lines of tangled code — three near-identical loop methods, inline shape discovery, scattered callback dispatch. It undermines the spike's thesis that JAX produces cleaner code than PyTorch, whose Trainer is 112 lines.

## Solution

Decompose into 4 files with a strategy protocol. Three loop implementations: PythonLoop (gymnasium), ScanLoop (gymnax scan), PmapLoop (multi-device parallel training).

## File layout

```
spike/trainer/
    __init__.py     # re-exports Trainer, PythonLoop, ScanLoop, PmapLoop, TrainingLoop
    _carry.py       # TrainCarry, StepOutput, TrainConfig
    _loops.py       # TrainingLoop protocol, PythonLoop, ScanLoop, PmapLoop,
                    #   _train_step, collect_batch, should_learn, shape discovery
    _trainer.py     # Trainer class
```

## Types

### TrainConfig

Frozen dataclass with slots bundling loop-invariant config. Replaces 11 kwargs on the protocol.

```python
@dataclass(frozen=True, slots=True)
class TrainConfig:
    num_steps: int
    checkpoint_steps: int
    collect_size: int
    min_buffer_size: int
    batch_size: int
    seed: int
    run_dir: Path | None
```

### TrainCarry

Monadic carry for lax.scan. Used by ScanLoop and PmapLoop. PythonLoop uses local variables.

```python
@chex.dataclass
class TrainCarry:
    agent_state: PyTree[Array]
    env_state: PyTree[Array]
    buffer: ExperienceBuffer
    step_count: Array      # int32 scalar
    key: PRNGKeyArray
```

### StepOutput

Per-step scan output for deferred callback dispatch. Internal to ScanLoop/PmapLoop.

```python
@chex.dataclass
class StepOutput:
    done: Array
    episode_return: Array
    episode_length: Array
    metrics: PyTree[Array]   # scalar metrics, keys fixed at trace time
    did_learn: Array
```

## TrainingLoop protocol

```python
class TrainingLoop(Protocol):
    def run(
        self,
        agent,
        env,
        *,
        initial_carry: TrainCarry,
        config: TrainConfig,
        callbacks: list,
    ) -> Any:
        """Execute the training loop. Returns final agent_state."""
        ...
```

## Shared predicates

Both PythonLoop and ScanLoop call these. `collect_batch` encapsulates the on-policy/off-policy distinction — the protocol never sees `on_policy`.

```python
def should_learn(steps_since_learn, buffer_size, collect_size, min_buffer_size):
    """Works with Python ints and JAX arrays."""
    return (steps_since_learn >= collect_size) & (buffer_size >= min_buffer_size)

def collect_batch(buffer, key, *, collect_size, batch_size):
    """Drain for on-policy (collect_size > 1), sample for off-policy."""
    if collect_size > 1:  # Python bool, resolved at trace time
        batch, _size, new_buffer = buffer_drain(buffer)
        return batch, new_buffer
    else:
        batch, _indices, _weights = buffer_sample(buffer, key, batch_size)
        return batch, buffer
```

## Loop implementations

### PythonLoop (~60 lines)

Python for-loop with JIT'd agent methods. Fires callbacks inline with full metrics. Does NOT use TrainCarry or `_train_step`. Works with any env (gymnasium or gymnax).

### ScanLoop (~80 lines)

`lax.scan` inner loop over checkpoint-sized segments. Shape discovery via `jax.eval_shape` (zero FLOPs). `_train_step` is the pure scan body with `lax.cond` for conditional learning. Callbacks fire at segment boundaries. Outputs all scalar metrics discovered at trace time.

### PmapLoop (~50 lines)

Multi-device parallel training via `jax.pmap`. Each device runs an independent ScanLoop with its own agent state, env state, and buffer. No synchronisation during collection — each device collects its own trajectories. Metrics are aggregated across devices at checkpoint boundaries via `jax.lax.pmean`.

```python
class PmapLoop:
    """Multi-device parallel training via jax.pmap.

    Each device runs an independent scan loop with its own agent state
    and environment. This is trivially parallel — no gradient synchronisation,
    no shared replay buffer — each device is a separate training run.

    At checkpoint boundaries, metrics are averaged across devices for logging.
    This is the pattern that would require multiprocessing + shared memory +
    serialisation in PyTorch.
    """

    def __init__(self, num_devices: int | None = None):
        self.num_devices = num_devices or jax.device_count()

    def run(self, agent, env, *, initial_carry, config, callbacks):
        # Replicate carry across devices
        carries = jax.device_put_replicated(initial_carry, jax.devices()[:self.num_devices])

        # Split keys across devices
        keys = jax.random.split(initial_carry.key, self.num_devices)

        # The scan body is the same _train_step used by ScanLoop
        # pmap wraps it: each device runs the scan independently
        def device_train(carry, key):
            carry = carry.replace(key=key)
            # Run ScanLoop's segment logic on this device
            ...
            return carry, metrics

        p_train = jax.pmap(device_train)

        for _seg in range(num_segments):
            carries, segment_metrics = p_train(carries, segment_keys)
            # Aggregate metrics across devices
            avg_metrics = jax.tree.map(lambda x: x.mean(axis=0), segment_metrics)
            # Fire callbacks with averaged metrics
            ...

        # Return first device's state (or all, depending on use case)
        return jax.tree.map(lambda x: x[0], carries).agent_state
```

The key insight: `pmap` over the scan body means N devices each run the full training loop independently. No communication during training, only metric aggregation at checkpoints. This is embarrassingly parallel and proves the TrainingLoop protocol works as an extension point.

## Trainer class (~90 lines)

```python
class Trainer:
    def __init__(self, agent, env, *, num_steps, checkpoint_steps,
                 action_shape=None, buffer_capacity=None, batch_size=32,
                 min_buffer_size=None, run_dir=None, callbacks=None,
                 seed=42, loop=None):
        # Config derivation (same as current)
        # Auto-detect action_shape via trial act()
        # Auto-select loop:
        #   - gymnasium env → PythonLoop
        #   - gymnax env → ScanLoop
        #   - user override via loop= kwarg

    def make_initial_state(self, key) -> TrainCarry:
        """Public factory for power users (checkpoint resume, warm-start)."""

    def fit(self, key, *, carry=None) -> agent_state:
        """Run training. Delegates to self._loop.run()."""
        if carry is None:
            carry = self.make_initial_state(key)
        return self._loop.run(
            self.agent, self.env,
            initial_carry=carry,
            config=self._config,
            callbacks=self.callbacks,
        )
```

## train_step (pure function, ~50 lines)

The single pure function used by ScanLoop and PmapLoop as the `lax.scan` body.

```python
def _train_step(carry, agent, env, *, config, zero_metrics, scalar_keys):
    # 1. act
    # 2. env.step
    # 3. capture terminal return/length before auto-reset
    # 4. make_transition, buffer_add
    # 5. lax.cond(should_learn, _do_learn, _skip_learn)
    #    where _do_learn calls collect_batch() + agent.learn()
    # 6. return (new_carry, StepOutput)
```

## Shape discovery

Via `jax.eval_shape` — traces agent.learn() without executing, zero FLOPs. Only called by ScanLoop/PmapLoop.

```python
def _discover_metrics_shape(agent, state, dummy_batch):
    """Trace agent.learn() to discover metrics pytree structure.

    Uses jax.eval_shape for zero-cost shape inference.
    Returns (zero_metrics, scalar_keys).
    """
    try:
        _, metrics_shapes = jax.eval_shape(agent.learn, state, dummy_batch, key)
        zero_metrics = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), metrics_shapes)
        scalar_keys = [k for k, v in metrics_shapes.items() if v.shape == ()]
    except Exception as e:
        raise RuntimeError(
            f"Shape discovery failed: {e}. Check agent.learn() works with zero inputs."
        ) from e
    return zero_metrics, scalar_keys
```

## Callback contract

Both loops produce `dict[str, float]` for `on_step`. PythonLoop includes all scalar metrics from agent.learn(). ScanLoop includes all scalar metrics discovered at trace time. Non-scalar metrics are PythonLoop-only — ScanLoop emits a one-time warning when dropping them.

This is an inherent JAX constraint (scan outputs must have fixed pytree structure). Documented in the Callback protocol docstring.

## Consumer API

```python
# Simple (auto-dispatch):
state = Trainer(agent, env, num_steps=100_000, checkpoint_steps=2500).fit(key)

# Force Python loop (debugging):
from spike.trainer import PythonLoop
state = Trainer(agent, env, ..., loop=PythonLoop()).fit(key)

# Multi-device parallel:
from spike.trainer import PmapLoop
state = Trainer(agent, env, ..., loop=PmapLoop(num_devices=4)).fit(key)

# Resume from checkpoint:
trainer = Trainer(agent, env, ...)
carry = trainer.make_initial_state(key)
carry = carry.replace(agent_state=loaded_checkpoint)
state = trainer.fit(key, carry=carry)
```

## Success criteria

1. All 190 existing tests pass (no regression)
2. All 8 e2e training tests pass
3. All 10 showcase examples run
4. PmapLoop PoC demonstrates multi-device training (or graceful single-device fallback)
5. Total Trainer package under ~400 lines
6. No file over ~200 lines
