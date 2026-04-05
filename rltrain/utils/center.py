import torch as T


def center(x: T.Tensor) -> T.Tensor:
    """Centers a tensor of values by subtracting its mean and dividing by its standard deviation +
    a very-small value for numerical stability.

    Parameters
    ----------
    ``x`` : ``Tensor``
        The tensor of values to normalise.

    Returns
    -------
    ``Tensor``
        A copy of the input tensor with normalised values.
    """
    return (x - x.mean()) / (x.std() + T.finfo(x.dtype).eps)
