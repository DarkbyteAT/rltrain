import dataclasses as dc
from collections.abc import Iterator
from typing import Generic, TypeVar


U = TypeVar("U")


@dc.dataclass(eq=True, frozen=True, init=True, repr=True)
class Trajectory(Generic[U]):
    state: U
    action: U
    reward: U
    next_state: U
    done: U

    def __iter__(self) -> Iterator[U]:
        return iter(dc.astuple(self))
