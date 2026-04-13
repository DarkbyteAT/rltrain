r"""Trainer — runs the RL training loop with callback hooks and env-strategy dispatch.

Auto-dispatches between three environment strategies based on ``env.capabilities``:

- **Python loop** (gymnasium fallback): agent methods JIT'd, env stepping is Python
- **vmap batch** (gymnax): ``jax.vmap(env.step)`` over num_envs, Python outer loop
- **scan rollout** (gymnax): ``lax.scan`` inner loop, Python at checkpoint boundaries

The Trainer is generic over the agent state type ``S`` and never inspects its fields.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from spike.buffer import (
    buffer_add,
    buffer_drain,
    buffer_sample,
    make_buffer,
)
from spike.transitions import make_transition


class _NoOpCallback:
    """Default callback that does nothing — satisfies the Callback protocol."""

    def on_train_start(self, config: dict, run_dir: Path | None) -> None:
        """No-op."""

    def on_step(self, step: int, metrics: dict[str, float]) -> None:
        """No-op."""

    def on_episode_end(self, episode: int, episode_return: float, episode_length: int) -> None:
        """No-op."""

    def on_checkpoint(self, step: int, agent_state: object, run_dir: Path | None) -> None:
        """No-op."""

    def on_train_end(self, agent_state: object, run_dir: Path | None) -> None:
        """No-op."""


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Runs the training loop with callback hooks.

    Auto-dispatches between three env strategies based on ``env.capabilities``:

    - Python loop (gymnasium fallback): agent methods JIT'd, env is Python
    - vmap batch (gymnax): ``jax.vmap(env.step)`` over num_envs, Python outer loop
    - scan rollout (gymnax): ``lax.scan`` inner loop, Python at checkpoint boundaries
    """

    def __init__(
        self,
        agent,
        env,
        *,
        num_steps: int,
        checkpoint_steps: int,
        buffer_capacity: int | None = None,
        batch_size: int = 32,
        min_buffer_size: int | None = None,
        run_dir: Path | None = None,
        callbacks: list | None = None,
        seed: int = 42,
    ):
        """Initialise the Trainer.

        Args:
            agent: An object satisfying the Agent protocol (init/learn/act/collect_size).
            env: A GymnaxEnv or GymnasiumEnv.
            num_steps: Total environment steps to run.
            checkpoint_steps: Steps between checkpoint callbacks.
            buffer_capacity: Replay buffer capacity. Defaults to ``collect_size``
                for on-policy (drain pattern) or 1000 for off-policy.
            batch_size: Batch size for off-policy sampling.
            min_buffer_size: Minimum buffer fill before learning starts.
                Defaults to ``batch_size`` for off-policy agents.
            run_dir: Optional directory for checkpoints.
            callbacks: List of Callback instances. Defaults to a single no-op.
            seed: RNG seed.
        """
        self.agent = agent
        self.env = env
        self.num_steps = num_steps
        self.checkpoint_steps = checkpoint_steps
        self.run_dir = run_dir
        self.callbacks = callbacks if callbacks is not None else [_NoOpCallback()]
        self.seed = seed
        self.batch_size = batch_size

        self.collect_size = getattr(agent, "collect_size", 1)
        collect_size = self.collect_size
        self._on_policy = collect_size > 1

        if buffer_capacity is None:
            self.buffer_capacity = collect_size if self._on_policy else 1000
        else:
            self.buffer_capacity = buffer_capacity

        if min_buffer_size is None:
            self.min_buffer_size = collect_size if self._on_policy else batch_size
        else:
            self.min_buffer_size = min_buffer_size

        if self.num_steps % self.checkpoint_steps != 0:
            warnings.warn(
                f"num_steps ({self.num_steps}) is not divisible by checkpoint_steps "
                f"({self.checkpoint_steps}). Scan strategy will run "
                f"{(self.num_steps // self.checkpoint_steps) * self.checkpoint_steps} steps.",
                stacklevel=2,
            )

    def fit(self, key: PRNGKeyArray):
        """Run the training loop. Auto-dispatches on env.capabilities."""
        caps = self.env.capabilities
        if caps.scan_rollout:
            return self._fit_scan(key)
        elif caps.vmap_batch:
            return self._fit_vmap(key)
        else:
            return self._fit_python(key)

    # ------------------------------------------------------------------
    # Strategy: Python loop (gymnasium fallback)
    # ------------------------------------------------------------------

    def _fit_python(self, key: PRNGKeyArray):
        """Train using a Python for-loop with JIT'd agent methods.

        Suitable for gymnasium envs that cannot be traced by JAX.
        """
        agent = self.agent
        env = self.env
        collect_size = self.collect_size

        key, k_init, k_env = jax.random.split(key, 3)
        state = agent.init(k_init)
        obs = env.reset(k_env)

        act_jit = eqx.filter_jit(agent.act)
        learn_jit = eqx.filter_jit(agent.learn)

        buffer = make_buffer(self.buffer_capacity, env.obs_shape, ())
        episode_count = 0
        episode_return = 0.0
        episode_length = 0
        steps_since_learn = 0

        config = {"num_steps": self.num_steps, "seed": self.seed}
        for cb in self.callbacks:
            cb.on_train_start(config, self.run_dir)

        for step in range(self.num_steps):
            key, k_act, k_learn = jax.random.split(key, 3)

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

            # Episode boundary
            if bool(done):
                for cb in self.callbacks:
                    cb.on_episode_end(episode_count, episode_return, episode_length)
                episode_count += 1
                episode_return = 0.0
                episode_length = 0

            # Learn step
            if steps_since_learn >= collect_size and int(buffer.size) >= self.min_buffer_size:
                if self._on_policy:
                    batch, _size, buffer = buffer_drain(buffer)
                else:
                    batch, _indices, _weights = buffer_sample(buffer, k_learn, self.batch_size)
                state, metrics = learn_jit(state, batch, k_learn)
                steps_since_learn = 0

                py_metrics = {k: float(v) for k, v in metrics.items()}
                for cb in self.callbacks:
                    cb.on_step(step, py_metrics)

            # Checkpoint
            if (step + 1) % self.checkpoint_steps == 0:
                for cb in self.callbacks:
                    cb.on_checkpoint(step + 1, state, self.run_dir)

            obs = next_obs

        for cb in self.callbacks:
            cb.on_train_end(state, self.run_dir)

        env.close()
        return state

    # ------------------------------------------------------------------
    # Strategy: vmap batch (gymnax, Python outer loop)
    # ------------------------------------------------------------------

    def _fit_vmap(self, key: PRNGKeyArray):
        """Train using vmapped env stepping with a Python outer loop.

        Same structure as the Python loop but env.step is pure-JAX and the
        agent acts on a single observation (no explicit vectorisation over
        num_envs since gymnax envs are single-instance).
        """
        return self._fit_gymnax_python_loop(key)

    # ------------------------------------------------------------------
    # Strategy: scan rollout (gymnax, lax.scan inner)
    # ------------------------------------------------------------------

    def _fit_scan(self, key: PRNGKeyArray):
        """Train using lax.scan for the inner loop.

        The outer Python loop fires checkpoint callbacks between segments.
        The inner scan loop collects transitions and learns.
        """
        agent = self.agent
        env = self.env
        collect_size = self.collect_size

        key, k_init, k_env = jax.random.split(key, 3)
        state = agent.init(k_init)
        env_state = env.reset(k_env)
        buffer = make_buffer(self.buffer_capacity, env.obs_shape, ())

        # Discover metrics pytree shape by tracing a single learn call.
        # This ensures _skip_learn returns the exact same structure as _do_learn,
        # regardless of which keys the agent's learn method returns.
        _dummy_batch, _dummy_sz, _dummy_buf = buffer_drain(
            buffer_add(
                buffer,
                make_transition(
                    obs=jnp.zeros(env.obs_shape),
                    action=jnp.array(0),
                    reward=jnp.array(0.0),
                    next_obs=jnp.zeros(env.obs_shape),
                    done=jnp.array(False),
                ),
            )
        )
        _dummy_state, dummy_metrics = agent.learn(state, _dummy_batch, jax.random.PRNGKey(0))
        zero_metrics = jax.tree.map(jnp.zeros_like, dummy_metrics)

        config = {"num_steps": self.num_steps, "seed": self.seed}
        for cb in self.callbacks:
            cb.on_train_start(config, self.run_dir)

        num_segments = self.num_steps // self.checkpoint_steps
        episode_count = 0

        def _scan_body(carry, _step_idx):
            """One step of collect + conditional learn inside lax.scan."""
            agent_state, es, buf, step_count, rng = carry
            rng, k_act, k_step, k_learn = jax.random.split(rng, 4)

            # Act
            action = agent.act(agent_state, es.obs, k_act)

            # Step env
            new_es = env.step(es, action, k_step)

            # Capture terminal return/length BEFORE auto-reset zeroes them.
            # The env accumulates episode_return and episode_length in new_es,
            # but resets them to 0 when done=True. The pre-reset values are
            # es.episode_return + new_es.reward and es.episode_length + 1.
            terminal_return = es.episode_return + new_es.reward
            terminal_length = es.episode_length + 1

            # Build transition
            transition = make_transition(
                obs=es.obs,
                action=action,
                reward=new_es.reward,
                next_obs=new_es.obs,
                done=new_es.done,
            )
            buf = buffer_add(buf, transition)
            step_count = step_count + 1

            # Conditional learn: every collect_size steps when buffer has enough data
            steps_mod = step_count % collect_size
            should_learn = (steps_mod == 0) & (buf.size >= self.min_buffer_size)

            def _do_learn(args):
                s, b, k = args
                if self._on_policy:
                    batch, _sz, b_empty = buffer_drain(b)
                    s_new, met = agent.learn(s, batch, k)
                    return s_new, b_empty, met
                else:
                    batch, _idx, _w = buffer_sample(b, k, self.batch_size)
                    s_new, met = agent.learn(s, batch, k)
                    return s_new, b, met

            def _skip_learn(args):
                s, b, _ = args
                return s, b, zero_metrics

            agent_state, buf, metrics = jax.lax.cond(
                should_learn,
                _do_learn,
                _skip_learn,
                (agent_state, buf, k_learn),
            )

            carry = (agent_state, new_es, buf, step_count, rng)
            step_out = (
                new_es.done,
                terminal_return,
                terminal_length,
                metrics["loss"],
                should_learn,
            )
            return carry, step_out

        scan_fn = _scan_body  # lax.scan traces the body — no separate JIT needed

        global_step = 0
        step_count_arr = jnp.array(0, dtype=jnp.int32)

        for _seg in range(num_segments):
            carry = (state, env_state, buffer, step_count_arr, key)
            carry, segment_out = jax.lax.scan(scan_fn, carry, jnp.arange(self.checkpoint_steps))
            state, env_state, buffer, step_count_arr, key = carry
            dones, ep_returns, ep_lengths, losses, did_learns = segment_out

            # Fire episode callbacks for completed episodes in this segment
            for i in range(self.checkpoint_steps):
                if bool(dones[i]):
                    for cb in self.callbacks:
                        cb.on_episode_end(
                            episode_count,
                            float(ep_returns[i]),
                            int(ep_lengths[i]),
                        )
                    episode_count += 1

            # Fire on_step for learn steps in this segment
            for i in range(self.checkpoint_steps):
                if bool(did_learns[i]):
                    py_metrics = {"loss": float(losses[i])}
                    for cb in self.callbacks:
                        cb.on_step(global_step + i, py_metrics)

            global_step += self.checkpoint_steps

            for cb in self.callbacks:
                cb.on_checkpoint(global_step, state, self.run_dir)

        for cb in self.callbacks:
            cb.on_train_end(state, self.run_dir)

        # gymnax envs are pure-JAX and don't hold external resources to close
        return state

    # ------------------------------------------------------------------
    # Shared: gymnax Python-loop (used by both vmap and plain gymnax)
    # ------------------------------------------------------------------

    def _fit_gymnax_python_loop(self, key: PRNGKeyArray):
        """Train with a gymnax env using a Python outer loop.

        Used for both the vmap and plain gymnax strategies when a Python-level
        loop is acceptable. Agent act/learn are JIT'd; env.step is pure-JAX.
        """
        agent = self.agent
        env = self.env
        collect_size = self.collect_size

        key, k_init, k_env = jax.random.split(key, 3)
        state = agent.init(k_init)
        env_state = env.reset(k_env)
        buffer = make_buffer(self.buffer_capacity, env.obs_shape, ())

        act_jit = eqx.filter_jit(agent.act)
        learn_jit = eqx.filter_jit(agent.learn)
        step_jit = jax.jit(env.step)

        episode_count = 0
        steps_since_learn = 0

        config = {"num_steps": self.num_steps, "seed": self.seed}
        for cb in self.callbacks:
            cb.on_train_start(config, self.run_dir)

        for step in range(self.num_steps):
            key, k_act, k_step, k_learn = jax.random.split(key, 4)

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

            # Episode boundary — check done *before* auto-reset overwrites metrics
            if bool(new_env_state.done):
                # At the done step, episode_return/length have already been reset
                # to 0 by the auto-reset in GymnaxEnv.step. The terminal values
                # are available via the reward and length accumulated up to done.
                # We track them from the env_state *before* the step that triggers
                # done, plus the final reward.
                ep_ret = float(env_state.episode_return + new_env_state.reward)
                ep_len = int(env_state.episode_length + 1)
                for cb in self.callbacks:
                    cb.on_episode_end(episode_count, ep_ret, ep_len)
                episode_count += 1

            # Learn step
            if steps_since_learn >= collect_size and int(buffer.size) >= self.min_buffer_size:
                if self._on_policy:
                    batch, _size, buffer = buffer_drain(buffer)
                else:
                    batch, _indices, _weights = buffer_sample(buffer, k_learn, self.batch_size)
                state, metrics = learn_jit(state, batch, k_learn)
                steps_since_learn = 0

                py_metrics = {k: float(v) for k, v in metrics.items()}
                for cb in self.callbacks:
                    cb.on_step(step, py_metrics)

            # Checkpoint
            if (step + 1) % self.checkpoint_steps == 0:
                for cb in self.callbacks:
                    cb.on_checkpoint(step + 1, state, self.run_dir)

            env_state = new_env_state

        for cb in self.callbacks:
            cb.on_train_end(state, self.run_dir)

        # gymnax envs are pure-JAX and don't hold external resources to close
        return state
