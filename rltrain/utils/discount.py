import torch as T


@T.no_grad()
def discount(xs: T.Tensor, dones: T.Tensor, factor: float) -> T.Tensor:
    """Computes the discounted sum of ``xs``, with a factor of ``factor``.

    Parameters
    ----------
    ``xs`` : ``Tensor``
        1D-array of values to compute discounted sum over.
    ``dones`` : ``Tensor``
        1D-array of flags for whether whether the episode terminated or not at that step.
    ``factor`` : ``float``
        Discount factor for multiplication at each discounting step.

    Returns
    -------
    ``Tensor``
        1D-array of discounted rewards for each timestep.
    """

    xs_disc = T.zeros_like(xs)
    acc = T.zeros(1).to(xs.device)

    for i, (x, done) in enumerate(zip(reversed(xs), reversed(dones), strict=True)):
        acc = x.detach().clone() + (~done * factor * acc)
        xs_disc[i] = acc.detach().clone()

    return xs_disc.flip(0)
