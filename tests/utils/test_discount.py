import pytest
import torch as T

from rltrain.utils.discount import discount


@pytest.mark.unit
def test_discount_no_dones():
    # rewards [1, 1, 1], gamma=0.99, no dones
    # expected: [1 + 0.99 + 0.99^2, 1 + 0.99, 1]
    xs = T.tensor([1.0, 1.0, 1.0])
    dones = T.tensor([False, False, False])
    gamma = 0.99
    out = discount(xs, dones, gamma)
    expected = T.tensor([1 + 0.99 + 0.99**2, 1 + 0.99, 1.0])
    assert T.allclose(out, expected, atol=1e-5)


@pytest.mark.unit
def test_discount_episode_boundary():
    # done=True at step 1 should reset the accumulator
    xs = T.tensor([1.0, 1.0, 1.0])
    dones = T.tensor([False, True, False])
    gamma = 0.99
    out = discount(xs, dones, gamma)
    # step 2: 1.0 (last step, no future)
    # step 1: 1.0 (done=True, accumulator resets — ~True = False, so acc = 1.0)
    # step 0: 1.0 + 0.99 * 1.0 = 1.99
    expected = T.tensor([1.99, 1.0, 1.0])
    assert T.allclose(out, expected, atol=1e-5)


@pytest.mark.unit
def test_discount_single_step():
    xs = T.tensor([5.0])
    dones = T.tensor([False])
    out = discount(xs, dones, 0.99)
    assert T.allclose(out, T.tensor([5.0]))


@pytest.mark.unit
def test_discount_zero_factor():
    xs = T.tensor([1.0, 2.0, 3.0])
    dones = T.tensor([False, False, False])
    out = discount(xs, dones, 0.0)
    assert T.allclose(out, xs)
