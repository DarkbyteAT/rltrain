"""Trajectory dataclass representing a single environment transition."""

import dataclasses as dc
from collections.abc import Iterator
from typing import Generic, TypeVar


U = TypeVar("U")


@dc.dataclass(eq=True, frozen=True, init=True, repr=True)
class Trajectory(Generic[U]):
    """Frozen dataclass representing a single ``(state, action, reward, next_state, done)`` transition.

    Generic over ``U`` — typically ``np.ndarray`` from ``MDP.step()`` or ``torch.Tensor`` after batching.
    Iteration yields the five fields in order, enabling ``zip(*memory)`` unpacking in agent ``load()`` methods.
    """

    state: U
    action: U
    reward: U
    next_state: U
    done: U

    def __iter__(self) -> Iterator[U]:
        """Yield fields in ``(state, action, reward, next_state, done)`` order."""
        return iter(dc.astuple(self))
