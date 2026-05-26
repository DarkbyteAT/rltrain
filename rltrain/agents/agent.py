r"""Agent protocol, shared training state, and base classes for pure-functional RL agents.

The Agent contract has three methods:

- ``init(key) \to S``: construct the initial training state
- ``learn(state, batch, key) \to (state, metrics)``: one optimisation step
- ``act(state, obs, key) \to action``: action selection for environment interaction

The Agent module itself is **static** — it holds hyperparameters, network
architecture, and the optimizer, but no array leaves.  All mutable state
(parameters, optimizer state, target parameters) lives in a ``TrainState``
pytree that flows through ``lax.scan`` as the carry.

This separation means:

- JIT traces the Agent once and compiles it as a constant.
- ``jax.grad`` composes through ``learn`` because the state is a pure pytree.
- The Trainer is generic over the state type ``S`` and never inspects its fields.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, TypeVar, runtime_checkable

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from rltrain.transitions import Transition


S = TypeVar("S")


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------


@chex.dataclass
class TrainState:
    r"""Canonical training state — sufficient for most agents.

    All fields are pytrees of arrays, making the entire structure a valid
    ``lax.scan`` carry and ``jax.grad`` target.

    For on-policy agents that have no target network, ``target_params``
    should be initialised as a zero-filled sentinel with the same tree
    structure as ``params`` (matching rltrain's convention for optional
    pytree fields under scan/vmap).

    Agents with additional mutable state (e.g. step counters, epsilon
    schedules) can define their own dataclass — the Trainer is generic
    over ``S`` and never inspects field names.
    """

    params: PyTree[Array]
    opt_state: PyTree[Array]
    target_params: PyTree[Array]


# ---------------------------------------------------------------------------
# Agent protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Agent(Protocol[S]):
    """Three-method contract for pure-functional RL agents.

    The Trainer calls ``init``, ``learn``, and ``act`` without knowing
    the concrete state type ``S``.  Structural subtyping means any
    ``eqx.Module`` with matching method signatures satisfies this
    protocol — no explicit inheritance required.

    The optional ``collect_size`` attribute tells the Trainer how many
    transitions to accumulate before calling ``learn``.  On-policy agents
    set this to their horizon size; off-policy agents default to 1.
    The Trainer reads it via ``getattr(agent, 'collect_size', 1)``.
    """

    def init(self, key: PRNGKeyArray) -> S:
        """Construct the initial training state (params, opt_state, etc.)."""
        ...

    def learn(
        self,
        state: S,
        batch: Transition,
        key: PRNGKeyArray,
    ) -> tuple[S, dict[str, Float[Array, ""]]]:
        """One optimisation step: compute loss, update params, return new state."""
        ...

    def act(self, state: S, obs: Float[Array, " d"], key: PRNGKeyArray) -> Array:
        """Select an action given the current state and observation."""
        ...

    def act_batch(self, state: S, obs: Float[Array, "N d"], key: PRNGKeyArray) -> Array:
        """Select actions for a batch of N observations.

        The default implementation (see :func:`default_act_batch`) vmaps
        ``act`` over the leading axis after splitting the PRNG key. Agents
        that benefit from a fused batched forward pass — e.g. avoiding a
        per-element ``eqx.combine`` reconstruction — may override.
        """
        ...


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def default_act_batch(agent, state, obs: Float[Array, "N d"], key: PRNGKeyArray) -> Array:
    """Default batched-act: vmap ``agent.act`` over the leading axis.

    ``state`` is **not** mapped (it's a per-call constant, not a per-env one),
    only ``obs`` and the per-element PRNG keys are. Use as the body of every
    agent's ``act_batch`` unless a custom fused implementation is provided.
    """
    keys = jax.random.split(key, obs.shape[0])
    return jax.vmap(agent.act, in_axes=(None, 0, 0))(state, obs, keys)


def gradient_step(
    loss_fn,
    params: PyTree[Array],
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
) -> tuple[PyTree[Array], optax.OptState, Float[Array, ""]]:
    r"""One gradient descent step — the reusable 3-line core.

    Computes ``\nabla_\theta \mathcal{L}(\theta)``, applies the optax
    optimizer, and returns updated parameters.  Called once by simple
    agents (VanillaPG), multiple times by multi-objective agents (SAC).

    Args:
        loss_fn: Callable ``params -> scalar loss``.  Must close over any
            non-parameter arguments (batch, target params, etc.).
        params: Trainable parameter pytree.
        opt_state: Current optimizer state.
        optimizer: An ``optax.GradientTransformation``.

    Returns:
        ``(new_params, new_opt_state, loss_value)``
    """
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss_val


def gradient_step_with_aux(
    loss_fn,
    params: PyTree[Array],
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
) -> tuple[PyTree[Array], optax.OptState, Float[Array, ""], dict]:
    r"""Like ``gradient_step`` but ``loss_fn`` returns ``(loss, aux_dict)``.

    Uses ``has_aux=True`` in ``eqx.filter_value_and_grad`` to propagate
    auxiliary data (e.g. TD errors for PER) alongside the loss without a
    second forward pass.

    Args:
        loss_fn: Callable ``params -> (scalar_loss, aux_dict)``.
        params: Trainable parameter pytree.
        opt_state: Current optimizer state.
        optimizer: An ``optax.GradientTransformation``.

    Returns:
        ``(new_params, new_opt_state, loss_value, aux_dict)``
    """
    (loss_val, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss_val, aux


def init_target_params(params: PyTree[Array]) -> PyTree[Array]:
    """Create target parameters as a copy of the online parameters.

    JAX arrays are immutable, so this is structurally an identity — but it
    documents the intent that target and online start identical.
    """
    return jax.tree.map(lambda p: p, params)


def zero_target_params(params: PyTree[Array]) -> PyTree[Array]:
    """Create zero-filled sentinel target parameters (for on-policy agents)."""
    return jax.tree.map(jnp.zeros_like, params)


def dqn_learn_step(
    loss_fn,
    state,
    optimizer: optax.GradientTransformation,
    target_rate: float,
    eps_end: float,
    eps_decay: float,
):
    """Shared DQN learn step: gradient descent + Polyak + epsilon decay.

    Used by VanillaDQN, DoubleDQN, and DistributionalDQN to avoid
    duplicating the Polyak averaging and epsilon decay logic.
    """
    from rltrain.agents.vanilla_dqn import DQNState

    new_params, new_opt_state, loss_val = gradient_step(
        loss_fn,
        state.params,
        state.opt_state,
        optimizer,
    )
    new_target = optax.incremental_update(
        new_params,
        state.target_params,
        target_rate,
    )
    new_eps = jnp.maximum(
        jnp.array(eps_end),
        state.epsilon - jnp.array(eps_decay),
    )
    new_state = DQNState(
        params=new_params,
        opt_state=new_opt_state,
        target_params=new_target,
        epsilon=new_eps,
    )
    return new_state, {"loss": loss_val}


# ---------------------------------------------------------------------------
# On-policy base class
# ---------------------------------------------------------------------------


class OnPolicyAgent(eqx.Module):
    """Base class for on-policy agents with shared init/learn/act.

    Subclasses define their own fields (critic, hyperparameters) and
    override ``_loss``.  PPO/SPO override ``learn`` for their epoch loop
    but inherit ``init`` and ``act``.

    The ``actor`` and ``action_head`` fields are typed as ``eqx.Module``
    so any user-defined backbone or head composes here. ``action_head`` is
    expected to satisfy the :class:`rltrain.heads.Head` protocol — i.e.
    a callable mapping a feature vector to a ``distreqx`` distribution
    over actions — but the protocol is not enforced at the dataclass
    level to keep custom heads frictionless.
    """

    # ClassVar is excluded from dataclass fields, so subclasses can add
    # non-default fields without violating ordering rules. Override in
    # subclasses to change the default collect size.
    collect_size: ClassVar[int] = 256

    actor: eqx.Module
    action_head: eqx.Module
    optimizer: optax.GradientTransformation = eqx.field(static=True)

    def init(self, key: PRNGKeyArray) -> TrainState:
        """Construct the initial training state."""
        params, _static = eqx.partition(self, eqx.is_array)
        opt_state = self.optimizer.init(params)
        target_params = zero_target_params(params)
        return TrainState(
            params=params,
            opt_state=opt_state,
            target_params=target_params,
        )

    def learn(
        self,
        state: TrainState,
        batch: Transition,
        _key: PRNGKeyArray,
    ) -> tuple[TrainState, dict[str, Float[Array, ""]]]:
        r"""One gradient step on the loss."""
        static = eqx.partition(self, eqx.is_array)[1]

        def loss_fn(params):
            agent = eqx.combine(params, static)
            return agent._loss(batch)

        new_params, new_opt_state, loss_val = gradient_step(
            loss_fn,
            state.params,
            state.opt_state,
            self.optimizer,
        )
        return TrainState(
            params=new_params,
            opt_state=new_opt_state,
            target_params=state.target_params,
        ), {"loss": loss_val}

    def act(
        self,
        state: TrainState,
        obs: Float[Array, " d"],
        key: PRNGKeyArray,
    ) -> Array:
        """Sample an action from the policy."""
        static = eqx.partition(self, eqx.is_array)[1]
        agent = eqx.combine(state.params, static)
        features = agent.actor(obs)
        dist = agent.action_head(features)
        return dist.sample(key)

    def act_batch(
        self,
        state: TrainState,
        obs: Float[Array, "N d"],
        key: PRNGKeyArray,
    ) -> Array:
        """Sample actions for a batch of N observations (default vmap)."""
        return default_act_batch(self, state, obs, key)

    def _loss(self, batch: Transition) -> Float[Array, ""]:
        """Compute the scalar loss. Override in subclasses."""
        raise NotImplementedError
