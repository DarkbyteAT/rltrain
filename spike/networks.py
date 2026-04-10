"""Simple MLP network — inlined for the spike (no toblox dependency)."""

import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray


class MLP(eqx.Module):
    """Multi-layer perceptron with ReLU activations.

    Wraps ``eqx.nn.MLP`` with orthogonal weight initialisation matching
    rltrain's convention.
    """

    net: eqx.nn.MLP

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width: int,
        depth: int,
        *,
        key: PRNGKeyArray,
    ):
        self.net = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width,
            depth=depth,
            key=key,
        )

    def __call__(self, x: Float[Array, " d"]) -> Float[Array, " out"]:
        return self.net(x)
