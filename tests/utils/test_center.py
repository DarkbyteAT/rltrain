import pytest
import torch as T

from rltrain.utils.center import center


EPS = T.finfo(T.float32).eps


@pytest.mark.unit
def test_center_mean_and_std():
    x = T.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    out = center(x)
    assert T.isclose(out.mean(), T.tensor(0.0), atol=EPS)
    assert T.isclose(out.std(), T.tensor(1.0), atol=EPS)


@pytest.mark.unit
def test_center_2d():
    x = T.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = center(x)
    assert T.isclose(out.mean(), T.tensor(0.0), atol=EPS)
    assert T.isclose(out.std(), T.tensor(1.0), atol=EPS)


@pytest.mark.unit
def test_center_single_element():
    x = T.tensor([5.0])
    out = center(x)
    # std of a single element is NaN (0 degrees of freedom with Bessel's
    # correction), so center() produces NaN. This is expected — center()
    # is only called on batches of returns/advantages, never single values.
    assert T.isnan(out).all()


@pytest.mark.unit
def test_center_large_values():
    x = T.tensor([1e6, 2e6, 3e6])
    out = center(x)
    assert T.isclose(out.mean(), T.tensor(0.0), atol=EPS)
    assert T.isclose(out.std(), T.tensor(1.0), atol=EPS)
