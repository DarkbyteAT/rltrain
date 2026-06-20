"""Training loop implementations — PythonLoop, ScanLoop, PmapLoop, MultiSeedScanLoop.

Four strategies behind a common ``TrainingLoop`` protocol. The Trainer
auto-selects based on env capabilities, or the user overrides via the
``loop=`` kwarg.

Shared predicates (``should_learn``, ``collect_batch``) encapsulate the
on-policy / off-policy distinction so the protocol never sees ``on_policy``.

The parallelisation axes are orthogonal: ``ScanLoop`` runs one agent in one
env; ``PmapLoop`` shards across devices; ``MultiSeedScanLoop`` vmaps a stack
of agents (each with its own init seed) over a single env on one device.
"""

from __future__ import annotations

import copy
from typing import Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from rltrain.agents.agent import Agent
from rltrain.buffer import buffer_add, buffer_drain, buffer_sample, buffer_update_priorities
from rltrain.callbacks import Callback
from rltrain.env import Env
from rltrain.trainer._carry import StepOutput, TrainCarry, TrainConfig
from rltrain.transitions import make_transition


# ---------------------------------------------------------------------------
# TrainingLoop protocol
# ---------------------------------------------------------------------------


class TrainingLoop(Protocol):
    """Strategy interface for training loop implementations."""

    def run(
        self,
        agent: Agent,
        env: Env,
        *,
        initial_carry: TrainCarry,
        config: TrainConfig,
        callbacks: list[Callback],
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


def collect_batch(
    buffer,
    key: PRNGKeyArray,
    *,
    collect_size: int,
    batch_size: int,
    prioritised: bool = False,
):
    """Drain for on-policy (collect_size > 1), sample for off-policy.

    The on-policy/off-policy distinction is derived from ``collect_size``
    at trace time (Python bool), so JAX tracing sees only one branch.

    Returns ``(batch, new_buffer)``. The batch's ``is_weights`` and
    ``indices`` fields carry PER state when ``prioritised=True`` on the
    sampling path; the drain path leaves them as the sentinel values
    populated by :func:`make_buffer`.
    """
    if collect_size > 1:  # Python bool, resolved at trace time
        batch, _size, new_buffer = buffer_drain(buffer)
        return batch, new_buffer
    else:
        batch, _indices, _weights = buffer_sample(buffer, key, batch_size, prioritised=prioritised)
        return batch, buffer


def _apply_per_updates_segment(
    buffer,
    sample_indices: Array,
    td_errors: Array,
    did_learn: Array,
):
    """Apply a segment's worth of PER priority updates to the buffer.

    ``sample_indices`` and ``td_errors`` have shape ``(checkpoint_steps,
    batch_size)``; ``did_learn`` has shape ``(checkpoint_steps,)``. Steps
    that didn't learn are written back to their existing priorities (a
    no-op masked by ``did_learn``). Runs entirely in XLA — no device-to-
    host transfer of per-sample arrays.
    """

    def per_step(buf, xs):
        idx, td, learned = xs
        new_priorities = jnp.where(learned, td, buf.priorities[idx])
        return buf.replace(priorities=buf.priorities.at[idx].set(new_priorities)), None

    buffer, _ = jax.lax.scan(per_step, buffer, (sample_indices, td_errors, did_learn))
    return buffer


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


def _discover_metrics_shape(
    agent: Agent, state: Any, dummy_batch: Any, *, batch_size: int
) -> tuple[dict, list[str], tuple[int, ...], bool]:
    """Trace ``agent.learn()`` to discover metrics pytree structure.

    Uses ``jax.eval_shape`` for zero-cost shape inference — no FLOPs.

    Returns:
        ``(zero_metrics, scalar_keys, indices_shape, has_td_errors)``.
        ``zero_metrics`` contains only the scalar-valued keys (suitable
        for threading through ``lax.scan``); per-sample arrays like
        ``td_errors`` are stripped out and surfaced separately via
        :class:`StepOutput`. ``indices_shape`` is the per-step shape
        carried by ``StepOutput.td_errors`` and
        ``StepOutput.sample_indices`` — always ``(batch_size,)`` so
        the scan carry has fixed shape regardless of whether PER is
        active. ``has_td_errors`` flags whether the agent emits the
        key at all; the ScanLoop uses it as a Python-side predicate to
        decide whether to do the PER segment-boundary update.
    """
    try:
        _, metrics_shapes = jax.eval_shape(agent.learn, state, dummy_batch, jax.random.key(0))
    except Exception as e:
        raise RuntimeError(f"Shape discovery failed: {e}. Check agent.learn() works with zero inputs.") from e

    scalar_keys = [k for k, v in metrics_shapes.items() if v.shape == ()]
    zero_metrics = {k: jnp.zeros(metrics_shapes[k].shape, metrics_shapes[k].dtype) for k in scalar_keys}

    has_td_errors = "td_errors" in metrics_shapes
    indices_shape: tuple[int, ...] = (batch_size,)

    return zero_metrics, scalar_keys, indices_shape, has_td_errors


# ---------------------------------------------------------------------------
# _train_step — pure scan body
# ---------------------------------------------------------------------------


def _train_step(
    carry: TrainCarry,
    _step_idx: Array,
    *,
    agent: Agent,
    env: Env,
    config: TrainConfig,
    zero_metrics: dict,
    indices_shape: tuple[int, ...],
) -> tuple[TrainCarry, StepOutput]:
    """One step of collect + conditional learn inside ``lax.scan``.

    Pure function: no side effects, no Python control flow over traced values.

    Per-sample ``td_errors`` and the sampled buffer ``indices`` are pulled
    out of the agent's metrics dict and surfaced via :class:`StepOutput`.
    The ScanLoop / PmapLoop dispatch picks them up at segment boundaries
    and writes them back into ``buffer.priorities`` — keeping the
    scan-carry metrics scalar and the priority update Python-side.
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

    # Sentinels for the no-learn branch and for agents that don't emit
    # per-sample td_errors. Shape and dtype must match the do-learn branch
    # exactly so lax.cond's two branches return matching pytree shapes.
    zero_td_errors = jnp.zeros(indices_shape, dtype=jnp.float32)
    zero_indices = jnp.zeros(indices_shape, dtype=jnp.int32)

    def _do_learn(args):
        s, b, k = args
        batch, b_new = collect_batch(
            b,
            k,
            collect_size=config.collect_size,
            batch_size=config.batch_size,
            prioritised=config.prioritised,
        )
        s_new, met = agent.learn(s, batch, k)
        # Pop td_errors out of the scan-threaded metrics dict; it travels on
        # StepOutput instead so the scan carry stays scalar-only.
        td = met.pop("td_errors", zero_td_errors)
        # Drop any remaining non-scalar keys so the carried metrics pytree
        # matches `zero_metrics` (scalar-only by construction).
        scalar_met = {k_: v for k_, v in met.items() if v.shape == ()}
        # Sample indices: take what the buffer populated. On-policy drain
        # paths produce the per-slot sentinel zeros; off-policy sample
        # populates real positions. The shape matches indices_shape
        # because both buffer paths align with batch_size.
        sample_idx = batch.indices.astype(jnp.int32)
        return s_new, b_new, scalar_met, td, sample_idx

    def _skip_learn(args):
        s, b, _ = args
        return s, b, zero_metrics, zero_td_errors, zero_indices

    agent_state, new_buffer, metrics, td_errors, sample_indices = jax.lax.cond(
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
        td_errors=td_errors,
        sample_indices=sample_indices,
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

    def run(
        self, agent: Agent, env: Env, *, initial_carry: TrainCarry, config: TrainConfig, callbacks: list[Callback]
    ) -> Any:
        """Execute the Python training loop."""
        pure_step = env.capabilities.pure_step

        key = initial_carry.key
        key, k_init = jax.random.split(key)
        state = initial_carry.agent_state
        buffer = initial_carry.buffer

        act_jit = eqx.filter_jit(agent.act)
        # Multi-env vectorisation: dispatch to act_batch when the env returns
        # a batched observation. ``act_batch`` is part of the Agent Protocol
        # — no defensive ``hasattr`` check; missing it is a contract violation
        # that will fail at jit-compile time with a clear error.
        act_batch_jit = eqx.filter_jit(agent.act_batch)
        learn_jit = eqx.filter_jit(agent.learn)

        def _act(state, obs, k):
            if obs.ndim > 1:
                return act_batch_jit(state, obs, k)
            return act_jit(state, obs, k)

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
        # Multi-env staging: collect (obs, action, reward, next_obs, done)
        # tuples per step, each of shape (num_envs, ...). Flushed in env-
        # contiguous order on learn boundaries so on-policy GAE doesn't leak
        # across env boundaries.
        pending_per_env_step: list = []

        cb_config = {"num_steps": config.num_steps, "seed": config.seed}
        for cb in callbacks:
            cb.on_train_start(cb_config, config.run_dir)

        for step in range(config.num_steps):
            key, k_act, k_step, k_learn = jax.random.split(key, 4)

            if pure_step:
                # Gymnax path
                action = _act(state, env_state.obs, k_act)
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
                action = _act(state, obs, k_act)
                next_obs, reward, done, _info = env.step(action)

                if obs.ndim > 1:
                    # Multi-env (num_envs > 1): accumulate per-env transition
                    # rows in a Python pending list, then flush in env-
                    # contiguous order so on-policy GAE per-env reshape inside
                    # the agent matches the buffer layout. Without this, GAE
                    # would bootstrap across env boundaries — see commit
                    # message for the bug history.
                    num_envs = obs.shape[0]
                    pending_per_env_step.append((obs, action, reward, next_obs, done))
                    steps_since_learn += num_envs

                    step_reward = float(jnp.sum(reward))
                    step_done = bool(jnp.any(done))
                    episode_return += step_reward
                    episode_length += 1

                    if step_done:
                        beta = getattr(env, "reward_run_rate", 0.1)
                        running_return = beta * episode_return + (1.0 - beta) * running_return
                        for cb in callbacks:
                            cb.on_episode_end(episode_count, episode_return, episode_length, running_return)
                        episode_count += 1
                        episode_return = 0.0
                        episode_length = 0
                        # Vector env auto-resets per-element; no manual reset needed.
                else:
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
                # Multi-env: flush the staged per-step rows into the buffer
                # in env-contiguous order BEFORE the drain. Each entry of
                # ``pending_per_env_step`` is a tuple of (obs, action,
                # reward, next_obs, done) with leading axis num_envs.
                # We add env 0's full trajectory first, then env 1's, etc.
                if pending_per_env_step:
                    pending_num_envs = pending_per_env_step[0][0].shape[0]
                    for e in range(pending_num_envs):
                        for obs_b, act_b, rew_b, nobs_b, done_b in pending_per_env_step:
                            buffer = buffer_add(
                                buffer,
                                make_transition(
                                    obs=obs_b[e],
                                    action=act_b[e],
                                    reward=rew_b[e],
                                    next_obs=nobs_b[e],
                                    done=done_b[e],
                                ),
                            )
                    pending_per_env_step = []

                batch, buffer = collect_batch(
                    buffer,
                    k_learn,
                    collect_size=config.collect_size,
                    batch_size=config.batch_size,
                    prioritised=config.prioritised,
                )
                state, metrics = learn_jit(state, batch, k_learn)
                steps_since_learn = 0

                # PER: route per-sample td_errors back into the buffer at the
                # original sample indices. ``batch.indices`` was populated by
                # ``buffer_sample`` (zero sentinels otherwise).
                if config.prioritised and "td_errors" in metrics:
                    buffer = buffer_update_priorities(buffer, batch.indices, metrics["td_errors"])

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

    def run(
        self, agent: Agent, env: Env, *, initial_carry: TrainCarry, config: TrainConfig, callbacks: list[Callback]
    ) -> Any:
        """Execute the scan-based training loop."""
        state = initial_carry.agent_state
        env_state = initial_carry.env_state
        buffer = initial_carry.buffer
        key = initial_carry.key

        # Shape discovery
        batch_size = config.batch_size if config.collect_size <= 1 else config.collect_size
        dummy_batch = _make_dummy_batch(
            obs_shape=env_state.obs.shape,
            action_shape=agent.act(state, env_state.obs, jax.random.key(0)).shape,
            batch_size=batch_size,
        )
        zero_metrics, scalar_keys, indices_shape, has_td_errors = _discover_metrics_shape(
            agent, state, dummy_batch, batch_size=batch_size
        )

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
                indices_shape=indices_shape,
            )

        global_step = 0
        per_active = config.prioritised and has_td_errors

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

            # PER priority update — done in-XLA at segment boundary so the
            # ``td_errors`` per-sample array never has to be lifted off
            # device. Vectorised over the segment via ``jax.vmap``; the
            # no-learn steps masked by ``did_learn`` write the existing
            # priorities back to themselves (no-op).
            if per_active:
                buffer = _apply_per_updates_segment(
                    buffer,
                    segment_out.sample_indices,
                    segment_out.td_errors,
                    segment_out.did_learn,
                )

            # One host transfer per segment — the Python loops below see
            # numpy arrays, so bool()/float()/int() are no-ops rather than
            # device-to-host syncs. With checkpoint_steps in the thousands
            # this is two orders of magnitude faster than per-index lifts.
            host_out = jax.device_get(segment_out)

            # Fire episode callbacks for completed episodes
            for i in range(config.checkpoint_steps):
                if bool(host_out.done[i]):
                    for cb in callbacks:
                        cb.on_episode_end(
                            episode_count,
                            float(host_out.episode_return[i]),
                            int(host_out.episode_length[i]),
                            float(host_out.running_return[i]),
                        )
                    episode_count += 1

            # Fire on_step for learn steps — extract scalar metrics only
            for i in range(config.checkpoint_steps):
                if bool(host_out.did_learn[i]):
                    py_metrics = {}
                    for k in scalar_keys:
                        py_metrics[k] = float(host_out.metrics[k][i])
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
        """Configure the loop with a device count.

        Args:
            num_devices: Number of XLA devices to shard across. ``None``
                uses ``jax.device_count()``. Falls back to :class:`ScanLoop`
                when ``num_devices <= 1``.
        """
        self.num_devices = num_devices or jax.device_count()

    def run(
        self, agent: Agent, env: Env, *, initial_carry: TrainCarry, config: TrainConfig, callbacks: list[Callback]
    ) -> Any:
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
            action_shape=agent.act(state, env_state.obs, jax.random.key(0)).shape,
            batch_size=batch_size,
        )
        zero_metrics, scalar_keys, indices_shape, has_td_errors = _discover_metrics_shape(
            agent, state, dummy_batch, batch_size=batch_size
        )

        cb_config = {"num_steps": config.num_steps, "seed": config.seed}
        for cb in callbacks:
            cb.on_train_start(cb_config, config.run_dir)

        num_segments = config.num_steps // config.checkpoint_steps
        num_devices = self.num_devices
        devices = jax.devices()[:num_devices]
        per_active = config.prioritised and has_td_errors

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
                    indices_shape=indices_shape,
                )

            return jax.lax.scan(scan_body, carry, jnp.arange(config.checkpoint_steps))

        p_segment = jax.pmap(_segment_scan, in_axes=(0, None))

        episode_count = 0
        global_step = 0

        for _seg in range(num_segments):
            carries, segment_out = p_segment(carries, None)

            # PER priority update — per-device, vmapped across the device
            # axis so each device's buffer ingests its own segment's
            # td_errors. Same XLA-side pattern as ScanLoop.
            if per_active:
                carries = carries.replace(
                    buffer=jax.vmap(_apply_per_updates_segment)(
                        carries.buffer,
                        segment_out.sample_indices,
                        segment_out.td_errors,
                        segment_out.did_learn,
                    )
                )

            # Average metrics across devices at checkpoint boundary
            avg_metrics = jax.tree.map(lambda x: x.mean(axis=0), segment_out.metrics)

            # Single host transfer for both segment_out and avg_metrics —
            # see ScanLoop for the rationale. Per-index float()/bool()/int()
            # calls in the Python loop below operate on numpy arrays.
            host_out = jax.device_get(segment_out)
            host_avg_metrics = jax.device_get(avg_metrics)

            # Fire episode callbacks (use first device's episode data)
            for i in range(config.checkpoint_steps):
                if bool(host_out.done[0, i]):
                    for cb in callbacks:
                        cb.on_episode_end(
                            episode_count,
                            float(host_out.episode_return[0, i]),
                            int(host_out.episode_length[0, i]),
                            float(host_out.running_return[0, i]),
                        )
                    episode_count += 1

            # Fire on_step with averaged metrics
            for i in range(config.checkpoint_steps):
                if bool(host_out.did_learn[0, i]):
                    py_metrics = {}
                    for k in scalar_keys:
                        py_metrics[k] = float(host_avg_metrics[k][i])
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


# ---------------------------------------------------------------------------
# MultiSeedScanLoop
# ---------------------------------------------------------------------------


class MultiSeedScanLoop:
    """``lax.scan`` inner loop ``eqx.filter_vmap``-ped over a seed axis.

    Sibling of :class:`ScanLoop` that runs ``n_seeds`` independently-initialised
    agents in parallel on a single device. The agent module and its
    ``TrainCarry`` arrive pre-stacked along a leading seed axis (the
    :class:`MultiSeedTrainer` orchestrates the stacking); this loop owns the
    vmap, the scan, the per-segment dispatch back to host, and the per-seed
    callback fan-out.

    Compared with :class:`PmapLoop` (which shards independent runs across
    devices), this strategy fits a single device by sharing the XLA graph
    across the seed axis — one compile covers all seeds, and FLOPs are
    batched into wider matmuls. Callbacks fire ``n_seeds`` times per segment
    boundary on the Python side, once per seed, with that seed's ``run_dir``
    of ``cfg.run_dir / f"seed_{i}"`` so artefacts don't collide.
    """

    def __init__(self, n_seeds: int) -> None:
        """Configure the loop with the size of the seed axis.

        Args:
            n_seeds: Length of the leading seed axis on all stacked inputs.
                Must match the leading axis of ``initial_carry.agent_state``,
                ``initial_carry.env_state``, ``initial_carry.buffer``,
                ``initial_carry.key``, and the ``agent`` module's array
                leaves. Each value of ``i`` in ``range(n_seeds)`` corresponds
                to one independent run.
        """
        if n_seeds < 1:
            raise ValueError(f"n_seeds must be >= 1, got {n_seeds}")
        self.n_seeds = n_seeds

    def run(
        self,
        agent: Agent,
        env: Env,
        *,
        initial_carry: TrainCarry,
        config: TrainConfig,
        callbacks: list[Callback],
    ) -> Any:
        """Execute the vmap-over-seeds training loop.

        Args:
            agent: Stack of ``n_seeds`` agents. Array leaves must carry a
                leading seed axis; static fields (hyperparameters) are
                broadcast by ``eqx.filter_vmap``.
            env: Single environment. Replicated implicitly across seeds via
                the vmap of ``env.step`` inside ``_train_step``.
            initial_carry: Stacked carry — every array leaf has leading
                axis ``n_seeds``. The buffer pytree is similarly stacked
                so each seed owns its own replay state.
            config: Frozen training config. ``run_dir`` is interpreted as
                the parent directory; per-seed subdirectories
                ``seed_{i}`` are passed to callbacks at segment boundaries.
            callbacks: Iterable of :class:`Callback`-shaped objects. Fired
                ``n_seeds`` times per segment boundary, once per seed, in
                seed order ``0, 1, ..., n_seeds - 1``.

        Returns:
            The final stacked agent state — leading axis ``n_seeds``.
            :class:`MultiSeedTrainer` unstacks this into a
            ``{seed_idx: TrainState}`` map.
        """
        n_seeds = self.n_seeds

        # Shape discovery — unstack one seed for the trial trace. The
        # per-seed agent + state pair must satisfy the same shape contract
        # ScanLoop sees, so we lift them via ``jax.tree.map`` (array leaves
        # only) and pass the first slice into the existing helpers.
        single_agent = _unstack_seed(agent, 0)
        single_state = _unstack_seed(initial_carry.agent_state, 0)
        single_env_state = _unstack_seed(initial_carry.env_state, 0)

        batch_size = config.batch_size if config.collect_size <= 1 else config.collect_size
        dummy_batch = _make_dummy_batch(
            obs_shape=single_env_state.obs.shape,
            action_shape=single_agent.act(single_state, single_env_state.obs, jax.random.key(0)).shape,
            batch_size=batch_size,
        )
        zero_metrics, scalar_keys, indices_shape, has_td_errors = _discover_metrics_shape(
            single_agent, single_state, dummy_batch, batch_size=batch_size
        )

        cb_config = {"num_steps": config.num_steps, "seed": config.seed, "n_seeds": n_seeds}
        seed_run_dirs = _seed_run_dirs(config.run_dir, n_seeds)
        # Per-seed dirs aren't created by the trainer; create them here so
        # callbacks that open files in ``on_train_start`` (CSVLogger,
        # Checkpoint) don't blow up on ``FileNotFoundError``. Mirrors the
        # behaviour of the CLI for the single-seed Trainer.
        for d in seed_run_dirs:
            if d is not None:
                d.mkdir(parents=True, exist_ok=True)
        # Built-in callbacks carry per-run state (open file handles,
        # accumulator lists). Sharing one instance across seeds would
        # clobber state on every ``on_train_start``. Deep-copy the list per
        # seed so the Callback protocol stays unchanged and each seed sees
        # an independent, stateful collaborator.
        per_seed_callbacks: list[list[Callback]] = [[copy.deepcopy(cb) for cb in callbacks] for _ in range(n_seeds)]
        for s in range(n_seeds):
            for cb in per_seed_callbacks[s]:
                cb.on_train_start(cb_config, seed_run_dirs[s])

        num_segments = config.num_steps // config.checkpoint_steps
        per_active = config.prioritised and has_td_errors

        def _per_seed_segment(seed_agent, seed_carry):
            def scan_body(carry, step_idx):
                return _train_step(
                    carry,
                    step_idx,
                    agent=seed_agent,
                    env=env,
                    config=config,
                    zero_metrics=zero_metrics,
                    indices_shape=indices_shape,
                )

            carry, segment_out = jax.lax.scan(scan_body, seed_carry, jnp.arange(config.checkpoint_steps))
            if per_active:
                carry = carry.replace(
                    buffer=_apply_per_updates_segment(
                        carry.buffer,
                        segment_out.sample_indices,
                        segment_out.td_errors,
                        segment_out.did_learn,
                    )
                )
            return carry, segment_out

        # filter_vmap handles static fields on the agent module; filter_jit
        # caches the compiled graph across segments. One compile covers all
        # n_seeds; subsequent segments reuse it.
        per_seed_segment_jit = eqx.filter_jit(eqx.filter_vmap(_per_seed_segment))

        carry = initial_carry
        episode_counts = [0] * n_seeds
        global_step = 0

        for _seg in range(num_segments):
            carry, segment_out = per_seed_segment_jit(agent, carry)

            # One host transfer per segment — segment_out leaves have shape
            # ``(n_seeds, checkpoint_steps, ...)``. Per-seed slicing below
            # touches numpy arrays so the inner loops don't sync per-index.
            host_out = jax.device_get(segment_out)

            for s in range(n_seeds):
                # Fire episode callbacks for completed episodes (this seed).
                for i in range(config.checkpoint_steps):
                    if bool(host_out.done[s, i]):
                        for cb in per_seed_callbacks[s]:
                            cb.on_episode_end(
                                episode_counts[s],
                                float(host_out.episode_return[s, i]),
                                int(host_out.episode_length[s, i]),
                                float(host_out.running_return[s, i]),
                            )
                        episode_counts[s] += 1

                # Fire on_step for learn steps — scalar metrics only.
                for i in range(config.checkpoint_steps):
                    if bool(host_out.did_learn[s, i]):
                        py_metrics = {k: float(host_out.metrics[k][s, i]) for k in scalar_keys}
                        for cb in per_seed_callbacks[s]:
                            cb.on_step(global_step + i, py_metrics)

            global_step += config.checkpoint_steps

            # Per-seed checkpoint with that seed's slice of the carry.
            for s in range(n_seeds):
                seed_state = _unstack_seed(carry.agent_state, s)
                for cb in per_seed_callbacks[s]:
                    cb.on_checkpoint(global_step, seed_state, seed_run_dirs[s])

        # Final per-seed train_end dispatch.
        for s in range(n_seeds):
            seed_state = _unstack_seed(carry.agent_state, s)
            for cb in per_seed_callbacks[s]:
                cb.on_train_end(seed_state, seed_run_dirs[s])

        return carry.agent_state


def _unstack_seed(stacked: Any, seed_idx: int) -> Any:
    """Slice the leading seed axis off every array leaf of a pytree.

    Uses ``eqx.partition`` to separate array from static leaves so the
    slice operation only touches arrays — non-array leaves (callables,
    Python primitives, ``optax`` transforms) pass through untouched.
    """
    arrays, static = eqx.partition(stacked, eqx.is_array)
    sliced_arrays = jax.tree.map(lambda x: x[seed_idx], arrays)
    return eqx.combine(sliced_arrays, static)


def _seed_run_dirs(run_dir, n_seeds: int):
    """Compute per-seed run_dir subpaths. ``None`` parent ⇒ all ``None``."""
    if run_dir is None:
        return [None] * n_seeds
    return [run_dir / f"seed_{i}" for i in range(n_seeds)]
