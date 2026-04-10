"""Experience buffer — unified pytree storage for all RL access patterns.

Three usage patterns from the same struct:
- Episode rollout (VanillaPG, REINFORCE): buffer_drain → use all → clear
- Horizon rollout (A2C, PPO, SPO): buffer_shuffle_into_minibatches → N epochs → clear
- Persistent replay (DQN, SAC): buffer_sample(uniform/prioritised) → FIFO eviction
"""

import chex
import jax
import jax.numpy as jnp

from spike.transitions import Transition


@chex.dataclass
class ExperienceBuffer:
    """Fixed-size circular buffer of transitions stored as a pytree of arrays.

    Args:
        data: Batched Transition with shape (capacity, ...) on each field.
        write_idx: Next write position (wraps at capacity).
        size: Number of valid entries (clamped at capacity).
        priorities: Optional priority weights for PER. Shape (capacity,).
    """

    data: Transition
    write_idx: chex.Array
    size: chex.Array
    priorities: chex.Array


def make_buffer(capacity: int, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]) -> ExperienceBuffer:
    """Allocate an empty buffer with pre-allocated arrays."""
    return ExperienceBuffer(
        data=Transition(
            obs=jnp.zeros((capacity, *obs_shape)),
            action=jnp.zeros((capacity, *action_shape)),
            reward=jnp.zeros(capacity),
            next_obs=jnp.zeros((capacity, *obs_shape)),
            done=jnp.zeros(capacity, dtype=jnp.bool_),
            log_prob=jnp.zeros(capacity),
            value=jnp.zeros(capacity),
        ),
        write_idx=jnp.array(0, dtype=jnp.int32),
        size=jnp.array(0, dtype=jnp.int32),
        priorities=jnp.ones(capacity),
    )


def buffer_add(buffer: ExperienceBuffer, transition: Transition) -> ExperienceBuffer:
    """Write a single transition at the current write position (circular)."""
    idx = buffer.write_idx
    capacity = buffer.data.obs.shape[0]

    new_data = jax.tree.map(lambda buf, t: buf.at[idx].set(t), buffer.data, transition)

    return buffer.replace(
        data=new_data,
        write_idx=(idx + 1) % capacity,
        size=jnp.minimum(buffer.size + 1, capacity),
        priorities=buffer.priorities.at[idx].set(1.0),
    )


def buffer_sample(
    buffer: ExperienceBuffer,
    key: chex.PRNGKey,
    batch_size: int,
    *,
    prioritised: bool = False,
) -> Transition:
    """Sample a batch of transitions (uniform or priority-weighted)."""
    capacity = buffer.data.obs.shape[0]
    if prioritised:
        # Normalise priorities over valid entries, zero-out invalid slots
        valid_priorities = jnp.where(
            jnp.arange(capacity) < buffer.size,
            buffer.priorities,
            0.0,
        )
        probs = valid_priorities / jnp.sum(valid_priorities)
        indices = jax.random.choice(key, capacity, shape=(batch_size,), replace=True, p=probs)
    else:
        indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=buffer.size)

    return jax.tree.map(lambda x: x[indices], buffer.data)


def buffer_shuffle_into_minibatches(
    buffer: ExperienceBuffer,
    key: chex.PRNGKey,
    minibatch_size: int,
) -> Transition:
    """Shuffle valid entries and reshape into scannable minibatches.

    Returns a Transition where each field has shape
    ``(num_minibatches, minibatch_size, ...)``, suitable for ``lax.scan``
    over the leading axis.
    """
    perm = jax.random.permutation(key, buffer.size)

    # Truncate to a multiple of minibatch_size
    num_minibatches = buffer.size // minibatch_size
    total = num_minibatches * minibatch_size
    perm = perm[:total]

    shuffled = jax.tree.map(lambda x: x[perm], buffer.data)
    return jax.tree.map(
        lambda x: x.reshape(num_minibatches, minibatch_size, *x.shape[1:]),
        shuffled,
    )


def buffer_drain(buffer: ExperienceBuffer) -> tuple[Transition, ExperienceBuffer]:
    """Return all valid data and an empty buffer (for on-policy use-once patterns)."""
    valid_data = jax.tree.map(lambda x: x[: buffer.size], buffer.data)
    empty = buffer.replace(
        write_idx=jnp.array(0, dtype=jnp.int32),
        size=jnp.array(0, dtype=jnp.int32),
    )
    return valid_data, empty


def buffer_update_priorities(
    buffer: ExperienceBuffer,
    indices: chex.Array,
    new_priorities: chex.Array,
) -> ExperienceBuffer:
    """Update priority weights at the given indices (for PER)."""
    return buffer.replace(priorities=buffer.priorities.at[indices].set(new_priorities))
