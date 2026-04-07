from typing import Any, TypeVar


XY = TypeVar("XY")


def lerp(input: XY, target: XY, step: float) -> XY:
    """Computes the linear interpolation between ``x`` and ``y``, with a step size of ``step``.

    Args:
        input: Starting value for interpolation.
        target: Ending value for interpolation.
        step: Size of step from 0 to 1, i.e. ``input`` to ``target``.
    """
    a: Any = input
    b: Any = target
    return ((1.0 - step) * a) + (step * b)
