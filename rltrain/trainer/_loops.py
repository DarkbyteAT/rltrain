"""Training loop implementations — PythonLoop, ScanLoop, PmapLoop.

Three strategies behind a common ``TrainingLoop`` protocol. The Trainer
auto-selects based on env capabilities, or the user overrides via the
``loop=`` kwarg.

Shared predicates (``should_learn``, ``collect_batch``) encapsulate the
on-policy / off-policy distinction so the protocol never sees ``on_policy``.
"""

from __future__ import annotations

import warnings
from typing import Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from rltrain.buffer import buffer_add, buffer_drain, buffer_sample
from rltrain.trainer._carry import StepOutput, TrainCarry, TrainConfig
from rltrain.transitions import make_transition


# ---------------------------------------------------------------------------
# TrainingLoop protocol
# ---------------------------------------------------------------------------


class TrainingLoop(Protocol):
    """Strategy interface for training loop implementations."""

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


# ---------------------------------------------------------------------------
# Shared predicates
# ---------------------------------------------------------------------------


def should_learn(
    steps_since_learn: int | Array,
    buffer_size: int | Array,
    collect_size: int,
    min_buffer_size: int,
) -> bool | Array:
    """Whether to trigger a learn step. Works with Python ints and JAX arrays."""
    return (steps_since_learn >= collect_size) & (buffer_size >= min_buffer_size)


def collect_batch(buffer, key: PRNGKeyArray, *, collect_size: int, batch_size: int):
    """Drain for on-policy (collect_size > 1), sample for off-policy.

    The on-policy/off-policy distinction is derived from ``collect_size``
    at trace time (Python bool), so JAX tracing sees only one branch.
    """
    if collect_size > 1:  # Python bool, resolved at trace time
        batch, _size, new_buffer = buffer_drain(buffer)
        return batch, new_buffer
    else:
        batch, _indices, _weights = buffer_sample(buffer, key, batch_size)
        return batch, buffer


# ---------------------------------------------------------------------------
# Shape discovery
# ---------------------------------------------------------------------------


def _make_dummy_batch(
    obs_shape: tuple[int, ...],
    action_shape: tuple[int, ...],
    batch_size: int,
):
    """Build a zero-filled batch with log_prob and value fields included."""
    return make_transition(
        obs=jnp.zeros((batch_size, *obs_shape)),
        action=jnp.zeros((batch_size, *action_shape)),
        reward=jnp.zeros(batch_size),
        next_obs=jnp.zeros((batch_size, *obs_shape)),
        done=jnp.zeros(batch_size, dtype=jnp.bool_),
        log_prob=jnp.zeros(batch_size),
        value=jnp.zeros(batch_size),
    )


def _discover_metrics_shape(agent, state, dummy_batch):
    """Trace ``agent.learn()`` to discover metrics pytree structure.

    Uses ``jax.eval_shape`` for zero-cost shape inference — no FLOPs.
    Returns ``(zero_metrics, scalar_keys)``.
    """
    try:
        _, metrics_shapes = jax.eval_shape(agent.learn, state, dummy_batch, jax.random.PRNGKey(0))
        zero_metrics = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), metrics_shapes)
        scalar_keys = [k for k, v in metrics_shapes.items() if v.shape == ()]
    except Exception as e:
        raise RuntimeError(f"Shape discovery failed: {e}. Check agent.learn() works with zero inputs.") from e

    non_scalar = [k for k, v in metrics_shapes.items() if v.shape != ()]
    if non_scalar:
        warnings.warn(
            f"ScanLoop dropping non-scalar metrics: {non_scalar}. Only scalar metrics are supported inside lax.scan.",
            stacklevel=3,
        )

    return zero_metrics, scalar_keys


# ---------------------------------------------------------------------------
# _train_step — pure scan body
# ---------------------------------------------------------------------------


def _train_step(carry: TrainCarry, _step_idx, *, agent, env, config, zero_metrics):
    """One step of collect + conditional learn inside ``lax.scan``.

    Pure function: no side effects, no Python control flow over traced values.
    """
    key, k_act, k_step, k_learn = jax.random.split(carry.key, 4)

    # 1. Act
    action = agent.act(carry.agent_state, carry.env_state.obs, k_act)

    # 2. Env step
    new_env_state = env.step(carry.env_state, action, k_step)

    # 3. Capture terminal return/length BEFORE auto-reset zeroes them
    terminal_return = carry.env_state.episode_return + new_env_state.reward
    terminal_length = carry.env_state.episode_length + 1

    # 4. Build transition, add to buffer
    transition = make_transition(
        obs=carry.env_state.obs,
        action=action,
        reward=new_env_state.reward,
        next_obs=new_env_state.obs,
        done=new_env_state.done,
    )
    new_buffer = buffer_add(carry.buffer, transition)
    new_step_count = carry.step_count + 1

    # 5. Conditional learn
    steps_mod = new_step_count % config.collect_size
    learn_flag = (steps_mod == 0) & (new_buffer.size >= config.min_buffer_size)

    def _do_learn(args):
        s, b, k = args
        batch, b_new = collect_batch(b, k, collect_size=config.collect_size, batch_size=config.batch_size)
        s_new, met = agent.learn(s, batch, k)
        return s_new, b_new, met

    def _skip_learn(args):
        s, b, _ = args
        return s, b, zero_metrics

    agent_state, new_buffer, metrics = jax.lax.cond(
        learn_flag,
        _do_learn,
        _skip_learn,
        (carry.agent_state, new_buffer, k_learn),
    )

    new_carry = TrainCarry(
        agent_state=agent_state,
        env_state=new_env_state,
        buffer=new_buffer,
        step_count=new_step_count,
        key=key,
    )
    step_out = StepOutput(
        done=new_env_state.done,
        episode_return=terminal_return,
        episode_length=terminal_length,
        running_return=new_env_state.running_return,
        metrics=metrics,
        did_learn=learn_flag,
    )
    return new_carry, step_out


# ---------------------------------------------------------------------------
# PythonLoop
# ---------------------------------------------------------------------------


class PythonLoop:
    """Python for-loop with JIT'd agent methods.

    Fires callbacks inline with full metrics. Works with any env
    (gymnasium or gymnax). Does NOT use TrainCarry or ``_train_step``.
    """

    def run(self, agent, env, *, initial_carry: TrainCarry, config: TrainConfig, callbacks: list) -> Any:
        """Execute the Python training loop."""
        pure_step = env.capabilities.pure_step

        key = initial_carry.key
        key, k_init = jax.random.split(key)
        state = initial_carry.agent_state
        buffer = initial_carry.buffer

        act_jit = eqx.filter_jit(agent.act)
        learn_jit = eqx.filter_jit(agent.learn)

        # Env init depends on capabilities
        if pure_step:
            env_state = initial_carry.env_state
            step_jit = jax.jit(env.step)
        else:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

        episode_count = 0
        episode_return = 0.0
        episode_length = 0
        running_return = 0.0
        steps_since_learn = 0

        cb_config = {"num_steps": config.num_steps, "seed": config.seed}
        for cb in callbacks:
            cb.on_train_start(cb_config, config.run_dir)

        for step in range(config.num_steps):
            key, k_act, k_step, k_learn = jax.random.split(key, 4)

            if pure_step:
                # Gymnax path
                action = act_jit(state, env_state.obs, k_act)
                new_env_state = step_jit(env_state, action, k_step)

                transition = make_transition(
                    obs=env_state.obs,
                    action=action,
                    reward=new_env_state.reward,
                    next_obs=new_env_state.obs,
                    done=new_env_state.done,
                )
                buffer = buffer_add(buffer, transition)
                steps_since_learn += 1

                if bool(new_env_state.done):
                    ep_ret = float(env_state.episode_return + new_env_state.reward)
                    ep_len = int(env_state.episode_length + 1)
                    ep_run = float(new_env_state.running_return)
                    for cb in callbacks:
                        cb.on_episode_end(episode_count, ep_ret, ep_len, ep_run)
                    episode_count += 1

                env_state = new_env_state
            else:
                # Gymnasium path
                action = act_jit(state, obs, k_act)
                next_obs, reward, done, _info = env.step(action)

                transition = make_transition(
                    obs=obs,
                    action=action,
                    reward=reward,
                    next_obs=next_obs,
                    done=done,
                )
                buffer = buffer_add(buffer, transition)
                steps_since_learn += 1

                episode_return += float(reward)
                episode_length += 1

                if bool(done):
                    beta = getattr(env, "reward_run_rate", 0.1)
                    running_return = beta * episode_return + (1.0 - beta) * running_return
                    for cb in callbacks:
                        cb.on_episode_end(episode_count, episode_return, episode_length, running_return)
                    episode_count += 1
                    episode_return = 0.0
                    episode_length = 0
                    next_obs = env.reset()

                obs = next_obs

            # Learn step
            if should_learn(steps_since_learn, int(buffer.size), config.collect_size, config.min_buffer_size):
                batch, buffer = collect_batch(
                    buffer,
                    k_learn,
                    collect_size=config.collect_size,
                    batch_size=config.batch_size,
                )
                state, metrics = learn_jit(state, batch, k_learn)
                steps_since_learn = 0

                py_metrics = {k: float(v) for k, v in metrics.items() if v.ndim == 0}
                for cb in callbacks:
                    cb.on_step(step, py_metrics)

            # Checkpoint
            if (step + 1) % config.checkpoint_steps == 0:
                for cb in callbacks:
                    cb.on_checkpoint(step + 1, state, config.run_dir)

        for cb in callbacks:
            cb.on_train_end(state, config.run_dir)

        if not pure_step:
            env.close()

        return state


# ---------------------------------------------------------------------------
# ScanLoop
# ---------------------------------------------------------------------------


class ScanLoop:
    """``lax.scan`` inner loop with Python checkpoint boundaries.

    Shape discovery via ``jax.eval_shape`` (zero FLOPs). ``_train_step``
    is the pure scan body. Callbacks fire at segment boundaries.
    """

    def run(self, agent, env, *, initial_carry: TrainCarry, config: TrainConfig, callbacks: list) -> Any:
        """Execute the scan-based training loop."""
        state = initial_carry.agent_state
        env_state = initial_carry.env_state
        buffer = initial_carry.buffer
        key = initial_carry.key

        # Shape discovery
        batch_size = config.batch_size if config.collect_size <= 1 else config.collect_size
        dummy_batch = _make_dummy_batch(
            obs_shape=env_state.obs.shape,
            action_shape=agent.act(state, env_state.obs, jax.random.PRNGKey(0)).shape,
            batch_size=batch_size,
        )
        zero_metrics, scalar_keys = _discover_metrics_shape(agent, state, dummy_batch)

        cb_config = {"num_steps": config.num_steps, "seed": config.seed}
        for cb in callbacks:
            cb.on_train_start(cb_config, config.run_dir)

        num_segments = config.num_steps // config.checkpoint_steps
        episode_count = 0
        step_count_arr = initial_carry.step_count

        def scan_body(carry, step_idx):
            return _train_step(
                carry,
                step_idx,
                agent=agent,
                env=env,
                config=config,
                zero_metrics=zero_metrics,
            )

        global_step = 0

        for _seg in range(num_segments):
            carry = TrainCarry(
                agent_state=state,
                env_state=env_state,
                buffer=buffer,
                step_count=step_count_arr,
                key=key,
            )
            carry, segment_out = jax.lax.scan(scan_body, carry, jnp.arange(config.checkpoint_steps))
            state = carry.agent_state
            env_state = carry.env_state
            buffer = carry.buffer
            step_count_arr = carry.step_count
            key = carry.key

            # Fire episode callbacks for completed episodes
            for i in range(config.checkpoint_steps):
                if bool(segment_out.done[i]):
                    for cb in callbacks:
                        cb.on_episode_end(
                            episode_count,
                            float(segment_out.episode_return[i]),
                            int(segment_out.episode_length[i]),
                            float(segment_out.running_return[i]),
                        )
                    episode_count += 1

            # Fire on_step for learn steps — extract scalar metrics only
            for i in range(config.checkpoint_steps):
                if bool(segment_out.did_learn[i]):
                    py_metrics = {}
                    for k in scalar_keys:
                        py_metrics[k] = float(segment_out.metrics[k][i])
                    for cb in callbacks:
                        cb.on_step(global_step + i, py_metrics)

            global_step += config.checkpoint_steps

            for cb in callbacks:
                cb.on_checkpoint(global_step, state, config.run_dir)

        for cb in callbacks:
            cb.on_train_end(state, config.run_dir)

        return state


# ---------------------------------------------------------------------------
# PmapLoop
# ---------------------------------------------------------------------------


class PmapLoop:
    """Multi-device parallel training via ``jax.pmap``.

    Each device runs an independent scan loop with its own agent state
    and environment. No synchronisation during collection — each device
    collects its own trajectories. Metrics are averaged across devices
    at checkpoint boundaries.
    """

    def __init__(self, num_devices: int | None = None):
        self.num_devices = num_devices or jax.device_count()

    def run(self, agent, env, *, initial_carry: TrainCarry, config: TrainConfig, callbacks: list) -> Any:
        """Execute multi-device parallel training."""
        if self.num_devices <= 1:
            return ScanLoop().run(
                agent,
                env,
                initial_carry=initial_carry,
                config=config,
                callbacks=callbacks,
            )

        state = initial_carry.agent_state
        env_state = initial_carry.env_state
        buffer = initial_carry.buffer
        key = initial_carry.key

        # Shape discovery (same as ScanLoop)
        batch_size = config.batch_size if config.collect_size <= 1 else config.collect_size
        dummy_batch = _make_dummy_batch(
            obs_shape=env_state.obs.shape,
            action_shape=agent.act(state, env_state.obs, jax.random.PRNGKey(0)).shape,
            batch_size=batch_size,
        )
        zero_metrics, scalar_keys = _discover_metrics_shape(agent, state, dummy_batch)

        cb_config = {"num_steps": config.num_steps, "seed": config.seed}
        for cb in callbacks:
            cb.on_train_start(cb_config, config.run_dir)

        num_segments = config.num_steps // config.checkpoint_steps
        num_devices = self.num_devices
        devices = jax.devices()[:num_devices]

        # Replicate carry across devices
        carry = TrainCarry(
            agent_state=state,
            env_state=env_state,
            buffer=buffer,
            step_count=initial_carry.step_count,
            key=key,
        )
        carries = jax.device_put_replicated(carry, devices)

        # Split keys across devices
        device_keys = jax.random.split(key, num_devices)
        carries = carries.replace(key=device_keys)

        def _segment_scan(carry, _unused):
            def scan_body(c, step_idx):
                return _train_step(
                    c,
                    step_idx,
                    agent=agent,
                    env=env,
                    config=config,
                    zero_metrics=zero_metrics,
                )

            return jax.lax.scan(scan_body, carry, jnp.arange(config.checkpoint_steps))

        p_segment = jax.pmap(_segment_scan, in_axes=(0, None))

        episode_count = 0
        global_step = 0

        for _seg in range(num_segments):
            carries, segment_out = p_segment(carries, None)

            # Average metrics across devices at checkpoint boundary
            avg_metrics = jax.tree.map(lambda x: x.mean(axis=0), segment_out.metrics)

            # Fire episode callbacks (use first device's episode data)
            for i in range(config.checkpoint_steps):
                if bool(segment_out.done[0, i]):
                    for cb in callbacks:
                        cb.on_episode_end(
                            episode_count,
                            float(segment_out.episode_return[0, i]),
                            int(segment_out.episode_length[0, i]),
                            float(segment_out.running_return[0, i]),
                        )
                    episode_count += 1

            # Fire on_step with averaged metrics
            for i in range(config.checkpoint_steps):
                if bool(segment_out.did_learn[0, i]):
                    py_metrics = {}
                    for k in scalar_keys:
                        py_metrics[k] = float(avg_metrics[k][i])
                    for cb in callbacks:
                        cb.on_step(global_step + i, py_metrics)

            global_step += config.checkpoint_steps

            # Checkpoint with first device's state
            first_state = jax.tree.map(lambda x: x[0], carries.agent_state)
            for cb in callbacks:
                cb.on_checkpoint(global_step, first_state, config.run_dir)

            # Split new keys for next segment
            new_keys = jax.random.split(carries.key[0], num_devices)
            carries = carries.replace(key=new_keys)

        for cb in callbacks:
            first_state = jax.tree.map(lambda x: x[0], carries.agent_state)
            cb.on_train_end(first_state, config.run_dir)

        return jax.tree.map(lambda x: x[0], carries.agent_state)
