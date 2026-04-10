"""Transition dataclass — the fundamental unit of RL experience."""

import chex
import jax.numpy as jnp


@chex.dataclass
class Transition:
    r"""A single environment transition $(s, a, r, s', d)$ with optional policy metadata.

    All fields are arrays. Fields unused by a particular agent (e.g. ``log_prob``
    for DQN) are zero-filled rather than None, because JAX requires fixed pytree
    structure under ``scan`` and ``vmap``.
    """

    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    next_obs: chex.Array
    done: chex.Array
    log_prob: chex.Array
    value: chex.Array


def make_transition(
    obs: chex.Array,
    action: chex.Array,
    reward: chex.Array,
    next_obs: chex.Array,
    done: chex.Array,
    log_prob: chex.Array | None = None,
    value: chex.Array | None = None,
) -> Transition:
    """Create a Transition, defaulting optional fields to zero scalars."""
    return Transition(
        obs=obs,
        action=action,
        reward=reward,
        next_obs=next_obs,
        done=done,
        log_prob=log_prob if log_prob is not None else jnp.zeros(()),
        value=value if value is not None else jnp.zeros(()),
    )
