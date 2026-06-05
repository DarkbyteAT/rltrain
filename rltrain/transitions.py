"""Transition dataclass — the fundamental unit of RL experience."""

import chex
import jax.numpy as jnp


@chex.dataclass
class Transition:
    r"""A single environment transition $(s, a, r, s', d)$ with optional policy metadata.

    All fields are arrays. Fields unused by a particular agent (e.g. ``log_prob``
    for DQN, or ``is_weights``/``indices`` for on-policy agents) are zero/one-filled
    rather than None, because JAX requires fixed pytree structure under ``scan``
    and ``vmap``.

    ``is_weights`` and ``indices`` carry PER state when an off-policy agent reads
    from a prioritised buffer. The uniform sampling path fills ``is_weights`` with
    ones (no IS correction) and ``indices`` with the sample positions so the trainer
    can route updated priorities back into the buffer after the learn step.
    """

    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    next_obs: chex.Array
    done: chex.Array
    log_prob: chex.Array
    value: chex.Array
    is_weights: chex.Array
    indices: chex.Array


def make_transition(
    obs: chex.Array,
    action: chex.Array,
    reward: chex.Array,
    next_obs: chex.Array,
    done: chex.Array,
    log_prob: chex.Array | None = None,
    value: chex.Array | None = None,
    is_weights: chex.Array | None = None,
    indices: chex.Array | None = None,
) -> Transition:
    """Create a Transition, defaulting optional fields to zero/one scalars."""
    return Transition(
        obs=obs,
        action=action,
        reward=reward,
        next_obs=next_obs,
        done=done,
        log_prob=log_prob if log_prob is not None else jnp.zeros(()),
        value=value if value is not None else jnp.zeros(()),
        is_weights=is_weights if is_weights is not None else jnp.ones(()),
        indices=indices if indices is not None else jnp.zeros((), dtype=jnp.int32),
    )
