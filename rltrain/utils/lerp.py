"""Linear interpolation utility for scalar and tensor values."""

from typing import TypeVar

import torch as T


XY = TypeVar("XY", float, T.Tensor)


def lerp(input: XY, target: XY, step: float) -> XY:
    """Computes the linear interpolation between ``input`` and ``target``, with a step size of ``step``.

    Args:
        input: Starting value for interpolation.
        target: Ending value for interpolation.
        step: Size of step from 0 to 1, i.e. ``input`` to ``target``.
    """
    return ((1.0 - step) * input) + (step * target)
