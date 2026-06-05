r"""Trainer -- orchestrates agent, env, buffer, and loop strategy.

The Trainer owns configuration derivation (buffer capacity, collect size,
action shape detection) and delegates the actual training loop to a
``TrainingLoop`` strategy selected from ``env.capabilities``.

Consumer API::

    # Simple (auto-dispatch):
    state = Trainer(agent, env, num_steps=100_000, checkpoint_steps=2500).fit(key)

    # Resume from checkpoint:
    trainer = Trainer(agent, env, ...)
    carry = trainer.make_initial_state(key)
    carry = carry.replace(agent_state=loaded_checkpoint)
    state = trainer.fit(key, carry=carry)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from rltrain.agents.agent import Agent
from rltrain.buffer import make_buffer
from rltrain.callbacks import Callback
from rltrain.env import Env
from rltrain.trainer._carry import TrainCarry, TrainConfig


if TYPE_CHECKING:
    from rltrain.trainer._loops import TrainingLoop


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


class Trainer:
    """Runs the training loop with callback hooks and env-strategy dispatch.

    Auto-selects a loop strategy from ``env.capabilities`` unless overridden
    via the ``loop`` kwarg.
    """

    def __init__(
        self,
        agent: Agent,
        env: Env,
        *,
        num_steps: int,
        checkpoint_steps: int,
        action_shape: tuple[int, ...] | None = None,
        buffer_capacity: int | None = None,
        batch_size: int = 32,
        min_buffer_size: int | None = None,
        run_dir: Path | None = None,
        callbacks: list[Callback] | None = None,
        seed: int = 42,
        loop: TrainingLoop | None = None,
        prioritised: bool = False,
    ) -> None:
        """Configure the trainer.

        Args:
            agent: Anything satisfying the :class:`rltrain.agents.Agent`
                protocol — ``init``, ``learn``, ``act``.
            env: An environment exposing ``EnvCapabilities`` — gymnax or
                gymnasium backed.
            num_steps: Total environment steps to run.
            checkpoint_steps: Steps between callback segment dispatches and
                checkpoint hooks.
            action_shape: Optional override for the action-space shape used
                to allocate the buffer. Auto-detected when ``None``.
            buffer_capacity: Replay capacity. Defaults to the agent's
                ``collect_size`` (on-policy) or a sensible default.
            batch_size: Mini-batch size drawn from the buffer per learn step.
            min_buffer_size: Warmup threshold before ``learn`` is called.
                Defaults to ``batch_size``.
            run_dir: Directory for callbacks (CSV/plots/checkpoints).
            callbacks: Iterable of :class:`Callback`-shaped objects. ``None``
                installs a single no-op callback.
            seed: PRNG seed used by ``fit`` when no key is supplied.
            loop: Explicit :class:`TrainingLoop`. ``None`` auto-selects from
                ``env.capabilities``.
            prioritised: Enable Prioritised Experience Replay. When ``True``,
                off-policy agents sample with priority weighting and the
                trainer writes ``td_errors`` back as new priorities after
                each learn step. No-op for on-policy agents.
        """
        self.agent = agent
        self.env = env
        self.num_steps = num_steps
        self.checkpoint_steps = checkpoint_steps
        self.run_dir = run_dir
        self.callbacks = callbacks if callbacks is not None else [_NoOpCallback()]
        self.seed = seed
        self.batch_size = batch_size

        # Derive collect_size from the agent (on-policy agents expose it).
        self.collect_size = getattr(agent, "collect_size", 1)
        self._on_policy = self.collect_size > 1

        # Auto-detect action shape via a trial act() call.
        if action_shape is None:
            self.action_shape = self._detect_action_shape(jax.random.key(seed))
        else:
            self.action_shape = action_shape

        # Buffer capacity defaults.
        if buffer_capacity is None:
            self.buffer_capacity = self.collect_size if self._on_policy else 1000
        else:
            self.buffer_capacity = buffer_capacity

        # Minimum buffer fill defaults.
        if min_buffer_size is None:
            self.min_buffer_size = self.collect_size if self._on_policy else batch_size
        else:
            self.min_buffer_size = min_buffer_size

        if num_steps % checkpoint_steps != 0:
            warnings.warn(
                f"num_steps ({num_steps}) is not divisible by checkpoint_steps "
                f"({checkpoint_steps}). Scan strategy will run "
                f"{(num_steps // checkpoint_steps) * checkpoint_steps} steps.",
                stacklevel=2,
            )

        # Build the frozen config bundle.
        self._config = TrainConfig(
            num_steps=num_steps,
            checkpoint_steps=checkpoint_steps,
            collect_size=self.collect_size,
            min_buffer_size=self.min_buffer_size,
            batch_size=batch_size,
            seed=seed,
            run_dir=run_dir,
            prioritised=prioritised,
        )

        # Auto-select loop strategy.
        if loop is not None:
            self._loop = loop
        elif env.capabilities.scan_rollout:
            from rltrain.trainer._loops import ScanLoop

            self._loop = ScanLoop()
        else:
            from rltrain.trainer._loops import PythonLoop

            self._loop = PythonLoop()

    # ------------------------------------------------------------------
    # Action-shape discovery
    # ------------------------------------------------------------------

    def _detect_action_shape(self, key: PRNGKeyArray) -> tuple[int, ...]:
        """Probe the agent's act() to discover the action shape for the buffer."""
        k_init, k_act = jax.random.split(key)
        state = self.agent.init(k_init)
        if hasattr(self.env, "reset"):
            if self.env.capabilities.pure_step:
                env_state = self.env.reset(k_act)
                obs = env_state.obs
            else:
                obs = self.env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
        else:
            obs = jnp.zeros(self.env.obs_shape)
        if obs.ndim > 1:
            # Batched env — strip the leading axis to probe a single-element action shape.
            action = self.agent.act(state, obs[0], k_act)
        else:
            action = self.agent.act(state, obs, k_act)
        return action.shape

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def make_initial_state(self, key: PRNGKeyArray) -> TrainCarry:
        """Build the initial ``TrainCarry`` for training or checkpoint resume."""
        key, k_init, k_env = jax.random.split(key, 3)
        agent_state = self.agent.init(k_init)

        if self.env.capabilities.pure_step:
            env_state = self.env.reset(k_env)
        else:
            env_state = self.env.reset()

        buffer = make_buffer(self.buffer_capacity, self.env.obs_shape, self.action_shape)

        return TrainCarry(
            agent_state=agent_state,
            env_state=env_state,
            buffer=buffer,
            step_count=jnp.array(0, dtype=jnp.int32),
            key=key,
        )

    def fit(self, key: PRNGKeyArray, *, carry: TrainCarry | None = None) -> Any:
        """Run training. Delegates to ``self._loop.run()``.

        Args:
            key: PRNG key for the training run.
            carry: Optional pre-built carry (for checkpoint resume).
                If ``None``, calls ``make_initial_state(key)``.

        Returns:
            The final agent state.
        """
        if carry is None:
            carry = self.make_initial_state(key)
        return self._loop.run(
            self.agent,
            self.env,
            initial_carry=carry,
            config=self._config,
            callbacks=self.callbacks,
        )
