"""Tests for the experience buffer primitives."""

import chex
import jax
import jax.numpy as jnp
import pytest

from rltrain.buffer import (
    buffer_add,
    buffer_add_batch,
    buffer_drain,
    buffer_sample,
    buffer_shuffle_into_minibatches,
    buffer_update_priorities,
    make_buffer,
)
from rltrain.transitions import make_transition


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


# ---------------------------------------------------------------------------
# buffer_add_batch — multi-env path
# ---------------------------------------------------------------------------


def _batched_transition(n: int, *, base: float = 0.0):
    """Build a (n, ...) batched transition with distinguishable obs rows."""
    return make_transition(
        obs=jnp.arange(n * 4, dtype=jnp.float32).reshape(n, 4) + base,
        action=jnp.arange(n),
        reward=jnp.arange(n, dtype=jnp.float32),
        next_obs=jnp.arange(n * 4, dtype=jnp.float32).reshape(n, 4) + 100 + base,
        done=jnp.zeros(n, dtype=jnp.bool_),
    )


@pytest.mark.unit
def test_buffer_add_batch_advances_cursor_by_n():
    """Given an empty buffer, add_batch with N transitions advances cursor to N and size to N."""
    buf = make_buffer(capacity=16, obs_shape=(4,), action_shape=())
    batch = _batched_transition(n=5)

    new_buf = buffer_add_batch(buf, batch)

    assert int(new_buf.write_idx) == 5
    assert int(new_buf.size) == 5


@pytest.mark.unit
def test_buffer_add_batch_wraps_around():
    """Given a buffer near capacity, add_batch wraps the write cursor correctly."""
    buf = make_buffer(capacity=4, obs_shape=(4,), action_shape=())
    batch = _batched_transition(n=6)

    new_buf = buffer_add_batch(buf, batch)

    # After writing 6 to a capacity-4 buffer: cursor at 6 mod 4 = 2, size clamped to 4.
    assert int(new_buf.write_idx) == 2
    assert int(new_buf.size) == 4


@pytest.mark.unit
def test_buffer_add_batch_preserves_data_order():
    """Sequential rows in the batch land in sequential buffer slots."""
    buf = make_buffer(capacity=8, obs_shape=(4,), action_shape=())
    batch = _batched_transition(n=3)

    new_buf = buffer_add_batch(buf, batch)

    chex.assert_trees_all_close(new_buf.data.obs[0], batch.obs[0])
    chex.assert_trees_all_close(new_buf.data.obs[1], batch.obs[1])
    chex.assert_trees_all_close(new_buf.data.obs[2], batch.obs[2])
    chex.assert_trees_all_close(new_buf.data.reward[:3], batch.reward)


@pytest.mark.unit
def test_buffer_add_batch_sequential_calls_accumulate():
    """Two add_batch calls in sequence advance the cursor cumulatively."""
    buf = make_buffer(capacity=16, obs_shape=(4,), action_shape=())
    batch_a = _batched_transition(n=3)
    batch_b = _batched_transition(n=4, base=1000.0)

    buf = buffer_add_batch(buf, batch_a)
    buf = buffer_add_batch(buf, batch_b)

    assert int(buf.write_idx) == 7
    assert int(buf.size) == 7
    # Slot 3 must be the first row of batch_b (not the first of batch_a).
    chex.assert_trees_all_close(buf.data.obs[3], batch_b.obs[0])
