"""Shared type aliases used across the rltrain package."""

from collections.abc import Container, Iterable
from pathlib import Path
from typing import TypeVar


T = TypeVar("T")

PathLike = str | Path
SupportsIn = Container[T] | Iterable[T]
