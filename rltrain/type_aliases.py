from pathlib import Path
from typing import Container, Iterable, TypeVar, Union

T = TypeVar("T")

PathLike = Union[str, Path]
SupportsIn = Union[Container[T], Iterable[T]]