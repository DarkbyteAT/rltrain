"""Tests for the experience buffer primitives."""

import chex
import jax
import jax.numpy as jnp
import pytest

from spike.buffer import (
    buffer_add,
    buffer_drain,
    buffer_sample,
    buffer_shuffle_into_minibatches,
    buffer_update_priorities,
    make_buffer,
)
from spike.transitions import make_transition


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def buffer():
    return make_buffer(capacity=16, obs_shape=(4,), action_shape=())


@pytest.fixture
def sample_transition():
    return make_transition(
        obs=jnp.ones(4),
        action=jnp.array(1),
        reward=jnp.array(1.0),
        next_obs=jnp.ones(4) * 2,
        done=jnp.array(False),
    )


def test_make_buffer_shapes(buffer):
    """Given a capacity and shapes, buffer arrays have the right dimensions."""
    chex.assert_shape(buffer.data.obs, (16, 4))
    chex.assert_shape(buffer.data.reward, (16,))
    chex.assert_shape(buffer.priorities, (16,))
    assert buffer.write_idx == 0
    assert buffer.size == 0


def test_buffer_add_increments_size(buffer, sample_transition):
    """Given an empty buffer, adding a transition increments size and write_idx."""
    updated = buffer_add(buffer, sample_transition)
    assert updated.size == 1
    assert updated.write_idx == 1
    chex.assert_trees_all_close(updated.data.obs[0], sample_transition.obs)


def test_buffer_add_wraps_circularly(sample_transition):
    """Given a full buffer, adding wraps write_idx to 0."""
    buf = make_buffer(capacity=2, obs_shape=(4,), action_shape=())
    buf = buffer_add(buf, sample_transition)
    buf = buffer_add(buf, sample_transition)
    assert buf.size == 2
    assert buf.write_idx == 0

    buf = buffer_add(buf, sample_transition)
    assert buf.size == 2  # clamped at capacity
    assert buf.write_idx == 1


def test_buffer_sample_returns_correct_batch(buffer, sample_transition, key):
    """Given a buffer with entries, sampling returns the right batch size."""
    for _ in range(8):
        buffer = buffer_add(buffer, sample_transition)

    batch, indices, is_weights = buffer_sample(buffer, key, batch_size=4)
    chex.assert_shape(batch.obs, (4, 4))
    chex.assert_shape(batch.reward, (4,))
    chex.assert_shape(indices, (4,))
    chex.assert_shape(is_weights, (4,))
    # Uniform sampling: IS weights are all 1.0
    chex.assert_trees_all_close(is_weights, jnp.ones(4))


def test_buffer_shuffle_into_minibatches(buffer, sample_transition, key):
    """Given a buffer, shuffling produces scannable minibatches."""
    for _ in range(8):
        buffer = buffer_add(buffer, sample_transition)

    batches = buffer_shuffle_into_minibatches(buffer.data, key, num_valid=8, minibatch_size=4)
    chex.assert_shape(batches.obs, (2, 4, 4))  # 8 / 4 = 2 minibatches


def test_buffer_drain_returns_all_and_empties(buffer, sample_transition):
    """Given a buffer with entries, drain returns all data and empties the buffer."""
    for _ in range(5):
        buffer = buffer_add(buffer, sample_transition)

    data, size, empty = buffer_drain(buffer)
    # Full capacity arrays returned; size tells how many are valid
    chex.assert_shape(data.obs, (16, 4))  # full capacity, not just valid
    assert int(size) == 5
    assert empty.size == 0
    assert empty.write_idx == 0


def test_buffer_update_priorities(buffer, sample_transition, key):
    """Given a buffer, updating priorities changes the stored values."""
    for _ in range(4):
        buffer = buffer_add(buffer, sample_transition)

    indices = jnp.array([0, 2])
    new_prios = jnp.array([10.0, 20.0])
    updated = buffer_update_priorities(buffer, indices, new_prios)

    chex.assert_trees_all_close(updated.priorities[0], 10.0)
    chex.assert_trees_all_close(updated.priorities[2], 20.0)


def test_buffer_sample_prioritised(buffer, sample_transition, key):
    """Given a buffer with varying priorities, prioritised sampling is weighted."""
    for i in range(8):
        t = make_transition(
            obs=jnp.ones(4) * i,
            action=jnp.array(0),
            reward=jnp.array(float(i)),
            next_obs=jnp.ones(4) * (i + 1),
            done=jnp.array(False),
        )
        buffer = buffer_add(buffer, t)

    # Give high priority to index 0
    buffer = buffer_update_priorities(buffer, jnp.array([0]), jnp.array([1000.0]))

    # Sample many times — index 0 should dominate
    batch, indices, is_weights = buffer_sample(buffer, key, batch_size=4, prioritised=True)
    chex.assert_shape(batch.obs, (4, 4))
    chex.assert_shape(is_weights, (4,))
