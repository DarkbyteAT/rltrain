import pytest
import torch as T

from rltrain.utils.lerp import lerp


@pytest.mark.unit
def test_lerp_step_zero_returns_input():
    assert lerp(10.0, 20.0, 0.0) == 10.0


@pytest.mark.unit
def test_lerp_step_one_returns_target():
    assert lerp(10.0, 20.0, 1.0) == 20.0


@pytest.mark.unit
def test_lerp_step_half_returns_midpoint():
    assert lerp(10.0, 20.0, 0.5) == 15.0


@pytest.mark.unit
def test_lerp_with_tensors():
    x = T.tensor([0.0, 2.0])
    y = T.tensor([10.0, 20.0])
    out = lerp(x, y, 0.5)
    expected = T.tensor([5.0, 11.0])
    assert T.allclose(out, expected)


@pytest.mark.unit
def test_lerp_negative_values():
    assert lerp(-10.0, 10.0, 0.5) == 0.0
