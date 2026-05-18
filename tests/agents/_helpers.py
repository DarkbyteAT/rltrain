"""Shared test helpers for agent tests.

Provides common constants and factory functions so individual test
modules only need to define agent-specific construction.
"""

import jax
import jax.numpy as jnp

from rltrain.transitions import Transition


OBS_DIM = 4
NUM_ACTIONS = 2
HIDDEN = 32
HORIZON = 32
MINIBATCH = 16


def _make_on_policy_transitions(key: jax.Array, n: int = 16) -> Transition:
    """Fabricate a batch of random transitions with (n,) action shape."""
    k1, k2, k3 = jax.random.split(key, 3)
    return Transition(
        obs=jax.random.normal(k1, (n, OBS_DIM)),
        action=jax.random.randint(k2, (n,), 0, NUM_ACTIONS),
        reward=jax.random.normal(k3, (n,)),
        next_obs=jax.random.normal(k1, (n, OBS_DIM)),
        done=jnp.zeros(n, dtype=jnp.bool_),
        log_prob=jnp.zeros(n),
        value=jnp.zeros(n),
    )


def _make_off_policy_batch(key: jax.Array, n: int = 32) -> Transition:
    """Fabricate a batch of random transitions for off-policy agents."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return Transition(
        obs=jax.random.normal(k1, (n, OBS_DIM)),
        action=jax.random.randint(k2, (n,), 0, NUM_ACTIONS),
        reward=jax.random.normal(k3, (n,)),
        next_obs=jax.random.normal(k4, (n, OBS_DIM)),
        done=jnp.zeros(n, dtype=jnp.bool_),
        log_prob=jnp.zeros(n),
        value=jnp.zeros(n),
    )
