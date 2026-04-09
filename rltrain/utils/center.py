"""Tensor whitening utility — zero mean, unit variance."""

import torch as T


def center(x: T.Tensor) -> T.Tensor:
    """Centers a tensor by subtracting the mean and dividing by std + epsilon.

    A very-small epsilon is added to the standard deviation for numerical stability.

    Args:
        x: The tensor of values to normalise.

    Returns:
        A copy of the input tensor with normalised values.
    """
    return (x - x.mean()) / (x.std() + T.finfo(x.dtype).eps)
