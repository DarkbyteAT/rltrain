r"""MultiSeedTrainer — N differently-initialised agents in parallel via ``jax.vmap``.

Sibling of :class:`rltrain.trainer.Trainer` that fits ``n_seeds`` independent
agent inits on a single device. Each seed gets its own initial parameters,
optimiser state, env state, buffer, and PRNG stream; they all share the XLA
graph (one compile) and execute as wider matmuls (one device, batched seed axis).

Consumer API::

    def make_agent(key):
        return PPO(actor=..., critic=..., action_head=..., ...)

    trainer = MultiSeedTrainer(
        make_agent, env,
        num_steps=200_000, n_seeds=5,
        callbacks=[CSVLoggerCallback(), PlotCallback(), CheckpointCallback()],
        run_dir="results/sweep/<env>/<arm>",
    )
    states = trainer.fit(jax.random.key(42))  # {seed: TrainState}

The factory replaces the single ``agent`` arg of :class:`Trainer`; everything
else mirrors the standard trainer API. Artefacts land under
``run_dir/seed_{i}/`` per seed; callbacks fire ``n_seeds`` times at each
segment boundary (once per seed, in seed order) with that seed's
``run_dir``. Composing with multi-device sharding is a separate strategy
(``MultiSeedPmapLoop`` is out of scope for this release — CPU/single-device
first).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from rltrain.agents.agent import Agent
from rltrain.buffer import make_buffer
from rltrain.callbacks import Callback
from rltrain.env import Env
from rltrain.trainer._carry import TrainCarry, TrainConfig
from rltrain.trainer._loops import MultiSeedScanLoop, _unstack_seed


class _NoOpCallback:
    """Default callback that does nothing -- satisfies the Callback protocol."""

    def on_train_start(self, config: dict, run_dir: Path | None) -> None:
        """No-op."""

    def on_step(self, step: int, metrics: dict[str, float]) -> None:
        """No-op."""

    def on_episode_end(
        self,
        episode: int,
        episode_return: float,
        episode_length: int,
        running_return: float = 0.0,
    ) -> None:
        """No-op."""

    def on_checkpoint(self, step: int, agent_state: object, run_dir: Path | None) -> None:
        """No-op."""

    def on_train_end(self, agent_state: object, run_dir: Path | None) -> None:
        """No-op."""


class MultiSeedTrainer:
    r"""Trainer that fits ``n_seeds`` agents in parallel via vmap over a seed axis.

    Mirrors :class:`Trainer`'s ergonomics but takes an ``agent_factory`` —
    a ``Callable[[PRNGKeyArray], Agent]`` — instead of a single ``agent``
    instance. The factory is called ``n_seeds`` times with distinct keys; the
    resulting agents are stacked along a leading seed axis via
    ``eqx.filter_vmap`` (static fields broadcast, array leaves stack), so the
    underlying :class:`MultiSeedScanLoop` can vmap one ``lax.scan`` body
    across all seeds.

    Compared with launching ``n_seeds`` separate :class:`Trainer.fit` calls
    sequentially, this strategy collapses ``n_seeds`` compile passes into
    one and amortises matmul launch overhead over the seed axis. It is the
    single-device counterpart of :class:`PmapLoop` (which shards across
    devices); the two compose orthogonally but multi-device support is a
    future strategy (``MultiSeedPmapLoop``).
    """

    def __init__(
        self,
        agent_factory: Callable[[PRNGKeyArray], Agent],
        env: Env,
        *,
        num_steps: int,
        n_seeds: int,
        checkpoint_steps: int | None = None,
        action_shape: tuple[int, ...] | None = None,
        buffer_capacity: int | None = None,
        batch_size: int = 32,
        min_buffer_size: int | None = None,
        run_dir: Path | str | None = None,
        callbacks: list[Callback] | None = None,
        seed: int = 42,
        prioritised: bool = False,
    ) -> None:
        """Configure the multi-seed trainer.

        Args:
            agent_factory: Callable ``key -> Agent``. Called ``n_seeds`` times
                inside the vmap closure; each invocation must produce a
                fresh agent with no shared array leaves across seeds.
            env: An environment exposing ``EnvCapabilities`` — must satisfy
                ``capabilities.scan_rollout=True`` (vmap+scan-compatible,
                i.e. a gymnax env). Replicated implicitly across seeds via
                ``jax.vmap(env.reset)`` and the vmap inside the loop.
            num_steps: Total environment steps per seed.
            n_seeds: Number of independent agent inits to run in parallel.
            checkpoint_steps: Steps between callback segment dispatches and
                checkpoint hooks. Defaults to ``num_steps`` (single segment).
            action_shape: Optional override for the buffer's action-space
                shape. Auto-detected from a single-seed probe when ``None``.
            buffer_capacity: Replay capacity per seed. Defaults to the
                agent's ``collect_size`` (on-policy) or 1000 (off-policy).
            batch_size: Mini-batch size drawn from the buffer per learn step.
            min_buffer_size: Warmup threshold before ``learn`` is called.
                Defaults to ``collect_size`` (on-policy) or ``batch_size``
                (off-policy).
            run_dir: Parent directory for callback artefacts. Per-seed
                subdirectories ``run_dir / seed_{i}`` are passed to
                callbacks; ``None`` propagates ``None`` to all seeds.
            callbacks: Iterable of :class:`Callback`-shaped objects. ``None``
                installs a single no-op callback. Fired ``n_seeds`` times
                per segment boundary — once per seed, in seed order — with
                that seed's ``run_dir``.
            seed: PRNG seed used by ``fit`` when no key is supplied.
            prioritised: Enable Prioritised Experience Replay per-seed.
                No-op for on-policy agents.

        Raises:
            ValueError: If ``n_seeds < 1`` or the environment does not
                expose ``capabilities.scan_rollout=True`` (the strategy
                relies on pure ``lax.scan`` rollout).
        """
        if n_seeds < 1:
            raise ValueError(f"n_seeds must be >= 1, got {n_seeds}")
        if not env.capabilities.scan_rollout:
            raise ValueError(
                "MultiSeedTrainer requires an env with scan_rollout=True "
                "(e.g. GymnaxEnv); got an env with capabilities="
                f"{env.capabilities}."
            )

        self.agent_factory = agent_factory
        self.env = env
        self.num_steps = num_steps
        self.n_seeds = n_seeds
        self.checkpoint_steps = checkpoint_steps if checkpoint_steps is not None else num_steps
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.callbacks = callbacks if callbacks is not None else [_NoOpCallback()]
        self.seed = seed
        self.batch_size = batch_size
        self.prioritised = prioritised

        # Probe one seed to derive collect_size, action shape, and buffer
        # defaults. The probe agent is discarded; the real per-seed agents
        # are built inside ``fit`` so the factory drives all randomness.
        probe_agent = agent_factory(jax.random.key(seed))
        self.collect_size = getattr(probe_agent, "collect_size", 1)
        self._on_policy = self.collect_size > 1

        if action_shape is None:
            self.action_shape = self._detect_action_shape(probe_agent, jax.random.key(seed))
        else:
            self.action_shape = action_shape

        if buffer_capacity is None:
            self.buffer_capacity = self.collect_size if self._on_policy else 1000
        else:
            self.buffer_capacity = buffer_capacity

        if min_buffer_size is None:
            self.min_buffer_size = self.collect_size if self._on_policy else batch_size
        else:
            self.min_buffer_size = min_buffer_size

        if self.checkpoint_steps > num_steps:
            # The loop runs `num_steps // checkpoint_steps` segments; if
            # checkpoint_steps > num_steps the integer division floors to
            # zero, the segment loop never enters, and `fit` would silently
            # return the initial state. Fail fast instead.
            raise ValueError(
                f"checkpoint_steps ({self.checkpoint_steps}) must be <= "
                f"num_steps ({num_steps}); otherwise no training segments "
                f"would run and `fit` would return the initial state."
            )
        if num_steps % self.checkpoint_steps != 0:
            warnings.warn(
                f"num_steps ({num_steps}) is not divisible by checkpoint_steps "
                f"({self.checkpoint_steps}). MultiSeedScanLoop will run "
                f"{(num_steps // self.checkpoint_steps) * self.checkpoint_steps} steps.",
                stacklevel=2,
            )

        self._config = TrainConfig(
            num_steps=num_steps,
            checkpoint_steps=self.checkpoint_steps,
            collect_size=self.collect_size,
            min_buffer_size=self.min_buffer_size,
            batch_size=batch_size,
            seed=seed,
            run_dir=self.run_dir,
            prioritised=prioritised,
        )

        self._loop = MultiSeedScanLoop(n_seeds=n_seeds)

    # ------------------------------------------------------------------
    # Action-shape discovery
    # ------------------------------------------------------------------

    def _detect_action_shape(self, probe_agent: Agent, key: PRNGKeyArray) -> tuple[int, ...]:
        """Probe the agent's ``act`` to discover the buffer's action shape.

        Strips the leading axis from ``obs`` only when the env is a real
        vector env (``num_envs > 1``). Plain ``obs.ndim > 1`` would
        misidentify gymnax envs with spatially-structured observations
        (e.g. MinAtar ``(10, 10, 4)``) as batched and call ``act`` with
        ``obs[0]`` — feeding a shape-``(10, 4)`` tensor to an encoder
        expecting the flattened single-env obs.
        """
        k_init, k_act, k_env = jax.random.split(key, 3)
        state = probe_agent.init(k_init)
        env_state = self.env.reset(k_env)
        obs = env_state.obs
        num_envs = getattr(self.env, "num_envs", 1)
        if num_envs > 1:
            action = probe_agent.act(state, obs[0], k_act)
        else:
            action = probe_agent.act(state, obs, k_act)
        return action.shape

    # ------------------------------------------------------------------
    # Initial-state construction (stacked)
    # ------------------------------------------------------------------

    def make_initial_state(self, key: PRNGKeyArray) -> tuple[Agent, TrainCarry]:
        r"""Build the stacked agent and ``TrainCarry`` for the vmapped loop.

        Splits ``key`` into ``2 * n_seeds + 1`` sub-keys — one per-seed pair
        for the agent factory and env reset, plus one fit-key seed for the
        per-seed scan body. The agent factory is invoked under
        ``eqx.filter_vmap`` so static fields broadcast and array leaves
        stack along the leading seed axis.

        Args:
            key: Master PRNG key. Deterministic given the same key.

        Returns:
            A tuple ``(agent_stack, carry)`` where ``agent_stack`` is the
            vmapped agent module and ``carry`` is a :class:`TrainCarry`
            with leading axis ``n_seeds`` on every array leaf.
        """
        build_keys, env_keys, fit_keys = self._split_master_key(key)

        # Build agent + init state in one vmapped pass so the seed axis lands
        # on agent params, opt_state, and any other array leaves the factory
        # produces. filter_vmap broadcasts static fields (hyperparameters).
        def _build_one(k_factory, k_init):
            seed_agent = self.agent_factory(k_factory)
            seed_state = seed_agent.init(k_init)
            return seed_agent, seed_state

        agent_stack, state_stack = eqx.filter_vmap(_build_one)(build_keys, fit_keys)

        # Env state: single-env vmap of reset gives ``n_seeds`` independent
        # initial trajectories.
        env_state_stack = jax.vmap(self.env.reset)(env_keys)

        # Buffers are pure pytrees of arrays — vmap of make_buffer over a
        # dummy axis stacks them along ``n_seeds``. Using a constant dummy
        # input keeps the closure pure.
        def _make_one_buffer(_dummy):
            return make_buffer(self.buffer_capacity, self.env.obs_shape, self.action_shape)

        buffer_stack = jax.vmap(_make_one_buffer)(jnp.arange(self.n_seeds))

        carry = TrainCarry(
            agent_state=state_stack,
            env_state=env_state_stack,
            buffer=buffer_stack,
            step_count=jnp.zeros(self.n_seeds, dtype=jnp.int32),
            key=fit_keys,
        )
        return agent_stack, carry

    def _split_master_key(self, key: PRNGKeyArray):
        """Deterministic key split: ``(build_keys, env_keys, fit_keys)``.

        Splits in a fixed order so two calls with the same master key
        produce the same per-seed sub-keys. Each of the three returned
        arrays has leading axis ``n_seeds``.
        """
        k_build_root, k_env_root, k_fit_root = jax.random.split(key, 3)
        build_keys = jax.random.split(k_build_root, self.n_seeds)
        env_keys = jax.random.split(k_env_root, self.n_seeds)
        fit_keys = jax.random.split(k_fit_root, self.n_seeds)
        return build_keys, env_keys, fit_keys

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, key: PRNGKeyArray) -> dict[int, Any]:
        """Run training for all seeds in parallel.

        Args:
            key: PRNG key for the run. Deterministic given the same key.

        Returns:
            A mapping ``{seed_idx: TrainState}`` from seed index to that
            seed's final agent state. Seed indices are ``0, 1, ..., n_seeds - 1``
            in the order produced by ``jax.random.split(key, n_seeds)``.
        """
        agent_stack, carry = self.make_initial_state(key)
        stacked_final_state = self._loop.run(
            agent_stack,
            self.env,
            initial_carry=carry,
            config=self._config,
            callbacks=self.callbacks,
        )
        # Delegate per-seed slicing to ``_unstack_seed`` so custom agent
        # states with non-array leaves (e.g. optax states carrying Python
        # primitives or callables) survive unstacking — plain ``jax.tree.map``
        # slicing would crash on those.
        return {i: _unstack_seed(stacked_final_state, i) for i in range(self.n_seeds)}
