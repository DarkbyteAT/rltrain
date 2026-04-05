from typing import TypeVar


XY = TypeVar("XY")


def lerp(input: XY, target: XY, step: float) -> XY:
    """Computes the linear interpolation between ``x`` and ``y``, with a step size of ``step``.

    Parameters
    ----------
    ``x``
        Starting value for interpolation.
    ``y``
        Ending value for interpolation.
    ``step``
        Size of step from 0 to 1, i.e. ``x`` to ``y``.
    """
    return ((1.0 - step) * input) + (step * target)
