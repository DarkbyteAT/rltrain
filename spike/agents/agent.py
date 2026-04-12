r"""Agent protocol and shared training state for pure-functional RL agents.

The Agent contract has three methods:

- ``init(key) \to S``: construct the initial training state
- ``learn(state, batch) \to (state, metrics)``: one optimisation step
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

from typing import Protocol, TypeVar, runtime_checkable

import chex
import equinox as eqx
import jax
import optax
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from spike.transitions import Transition


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
    """

    def init(self, key: PRNGKeyArray) -> S:
        """Construct the initial training state (params, opt_state, etc.)."""
        ...

    def learn(self, state: S, batch: Transition) -> tuple[S, dict[str, Float[Array, ""]]]:
        """One optimisation step: compute loss, update params, return new state."""
        ...

    def act(self, state: S, obs: Float[Array, " d"], key: PRNGKeyArray) -> Array:
        """Select an action given the current state and observation."""
        ...


# ---------------------------------------------------------------------------
# Shared utility: gradient step
# ---------------------------------------------------------------------------


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


def init_target_params(params: PyTree[Array]) -> PyTree[Array]:
    """Create target parameters as a copy of the online parameters."""
    return jax.tree.map(lambda p: p.copy(), params)


def zero_target_params(params: PyTree[Array]) -> PyTree[Array]:
    """Create zero-filled sentinel target parameters (for on-policy agents)."""
    return jax.tree.map(jax.numpy.zeros_like, params)
