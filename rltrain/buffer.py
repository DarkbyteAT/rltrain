"""Experience buffer — unified pytree storage for all RL access patterns.

Three usage patterns from the same struct:
- Episode rollout (VanillaPG, REINFORCE): buffer_drain → use all → clear
- Horizon rollout (A2C, PPO, SPO): buffer_shuffle_into_minibatches → N epochs → clear
- Persistent replay (DQN, SAC): buffer_sample(uniform/prioritised) → FIFO eviction
"""

import chex
import jax
import jax.numpy as jnp

from rltrain.transitions import Transition


@chex.dataclass
class ExperienceBuffer:
    """Fixed-size circular buffer of transitions stored as a pytree of arrays.

    Attributes:
        data: Batched :class:`Transition` with shape ``(capacity, ...)``
            on each field. The ``is_weights`` and ``indices`` slots are
            populated per-batch by :func:`buffer_sample` (uniform
            sampling fills them with sentinels); they're carried on the
            stored Transition so PER state never has to be threaded as
            a separate return value.
        write_idx: Next write position (wraps at capacity).
        size: Number of valid entries (clamped at capacity).
        priorities: Per-slot priority weights for PER. Shape
            ``(capacity,)``. Always allocated; the uniform sampling path
            leaves them at their default of 1.0.
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
            # is_weights/indices are populated per-batch by buffer_sample;
            # the per-slot storage carries sentinel values so the pytree
            # shape stays uniform.
            is_weights=jnp.ones(capacity),
            indices=jnp.zeros(capacity, dtype=jnp.int32),
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


def buffer_add_batch(buffer: ExperienceBuffer, batched_transition: Transition) -> ExperienceBuffer:
    """Write ``N`` sequential transitions where ``N`` is the leading batch dim.

    The cursor advances by ``N`` (mod capacity) and the size grows by at most ``N``.

    Implementation: ``jax.lax.scan`` over the env axis. Each step is a single
    ``buffer_add`` call, so the buffer's write_idx / size / priorities all
    stay coherent through wrap-around. Slow only if you naively pythonise it;
    inside jit/scan this fuses into a single XLA kernel.

    ``Transition`` carries zero-dim sentinel leaves (``log_prob``, ``value``,
    ``is_weights``, ``indices``) when the caller didn't supply them — ``lax.scan``
    can't slice a scalar along axis 0. Infer ``N`` from a leaf with leading dim
    (``obs``) and broadcast the scalar leaves up to ``(N, ...)`` so every leaf
    has a consistent scan axis.
    """
    n = batched_transition.obs.shape[0]

    def _ensure_batched(x):
        if x.ndim == 0:
            return jnp.broadcast_to(x, (n,))
        # Already batched (leading dim == n) — leave untouched.
        return x

    aligned = jax.tree.map(_ensure_batched, batched_transition)

    def add_one(buf: ExperienceBuffer, single_trans: Transition) -> tuple[ExperienceBuffer, None]:
        return buffer_add(buf, single_trans), None

    final_buf, _ = jax.lax.scan(add_one, buffer, aligned)
    return final_buf


def buffer_sample(
    buffer: ExperienceBuffer,
    key: chex.PRNGKey,
    batch_size: int,
    *,
    prioritised: bool = False,
    alpha: float = 0.6,
    beta: float = 0.4,
) -> tuple[Transition, chex.Array, chex.Array]:
    r"""Sample a batch of transitions (uniform or priority-weighted).

    Returns:
        ``(batch, indices, is_weights)`` where ``is_weights`` has shape
        ``(batch_size,)`` and provides importance-sampling correction.

        For uniform sampling, ``is_weights`` is all ones (no correction).
        For prioritised sampling:

        $$P(i) = p_i^\alpha / \sum_j p_j^\alpha$$
        $$w_i = (N \cdot P(i))^{-\beta} / \max_j w_j$$

    Args:
        buffer: Experience buffer to sample from.
        key: PRNG key.
        batch_size: Number of transitions to sample.
        prioritised: Whether to use priority-weighted sampling.
        alpha: Priority exponent — controls how much prioritisation is used.
            0 = uniform, 1 = full prioritisation.
        beta: IS correction exponent — controls how much bias correction.
            0 = no correction, 1 = full correction.
    """
    capacity = buffer.data.obs.shape[0]
    if prioritised:
        # Normalise priorities over valid entries, zero-out invalid slots.
        # The +1e-6 epsilon on valid slots guards against the cold-start case
        # where every td_error happens to be 0 — without it, ``powered.sum()``
        # would also be 0 and ``probs`` would be NaN.
        valid_priorities = jnp.where(
            jnp.arange(capacity) < buffer.size,
            buffer.priorities + 1e-6,
            0.0,
        )
        powered = valid_priorities**alpha
        probs = powered / jnp.sum(powered)
        indices = jax.random.choice(key, capacity, shape=(batch_size,), replace=True, p=probs)

        # IS weights: w_i = (N * P(i))^(-beta), normalised so max = 1
        sampled_probs = probs[indices]
        weights = (buffer.size * sampled_probs) ** (-beta)
        is_weights = weights / jnp.max(weights)
    else:
        indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=buffer.size)
        is_weights = jnp.ones(batch_size)

    batch = jax.tree.map(lambda x: x[indices], buffer.data)
    # Embed the per-batch IS weights and sample indices into the Transition
    # itself. Agents that don't use PER ignore these fields; the trainer
    # reads them post-learn to route updated priorities back into the buffer.
    batch = batch.replace(is_weights=is_weights, indices=indices)
    return batch, indices, is_weights


def buffer_shuffle_into_minibatches(
    arrays,
    key: chex.PRNGKey,
    num_valid: int,
    minibatch_size: int,
):
    """Shuffle valid entries of a pytree of arrays and reshape into scannable minibatches.

    Accepts any pytree whose leaves share a leading axis of length at
    least ``num_valid`` — a bare :class:`Transition`, a tuple of
    ``Transition``-plus-aux-arrays, a flat dict, etc. The shuffle
    permutation is shared across all leaves so per-element correspondence
    is preserved (action[i] still pairs with obs[i] post-reshape).

    ``num_valid`` must be a compile-time constant (which it is for
    horizon-based PPO where the horizon is a hyperparameter). This avoids
    dynamic shapes under JIT.

    Args:
        arrays: Pytree of arrays sharing a leading axis.
        key: PRNG key for shuffling.
        num_valid: Number of valid entries (must be static under JIT).
        minibatch_size: Size of each minibatch.

    Returns:
        The same pytree shape with each leaf reshaped to
        ``(num_minibatches, minibatch_size, ...)``, suitable for
        ``lax.scan`` over the leading axis.
    """
    perm = jax.random.permutation(key, num_valid)

    # Truncate to a multiple of minibatch_size
    num_minibatches = num_valid // minibatch_size
    total = num_minibatches * minibatch_size
    perm = perm[:total]

    def _shuffle_reshape(x):
        # Scalar sentinel leaves (e.g. zero-dim ``log_prob`` or ``is_weights``
        # placeholders from on-policy Transitions) have no leading axis to
        # permute; broadcast them to the per-minibatch shape so downstream
        # scan bodies see a consistent leaf structure.
        if x.ndim == 0:
            return jnp.broadcast_to(x, (num_minibatches, minibatch_size))
        return x[perm].reshape(num_minibatches, minibatch_size, *x.shape[1:])

    return jax.tree.map(_shuffle_reshape, arrays)


def buffer_drain(buffer: ExperienceBuffer) -> tuple[Transition, chex.Array, ExperienceBuffer]:
    """Return all valid data and an emptied buffer.

    Returns the full capacity array — the caller uses the returned ``size``
    to know how many entries are valid.  This avoids dynamic slicing under JIT.

    Returns:
        ``(data, size, empty_buffer)``
    """
    empty = buffer.replace(
        write_idx=jnp.array(0, dtype=jnp.int32),
        size=jnp.array(0, dtype=jnp.int32),
    )
    return buffer.data, buffer.size, empty


def buffer_update_priorities(
    buffer: ExperienceBuffer,
    indices: chex.Array,
    new_priorities: chex.Array,
) -> ExperienceBuffer:
    """Update priority weights at the given indices (for PER)."""
    return buffer.replace(priorities=buffer.priorities.at[indices].set(new_priorities))
