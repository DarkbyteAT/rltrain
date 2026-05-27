"""Simple MLP network with orthogonal weight initialisation."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


def _apply_orthogonal(mlp: eqx.nn.MLP, key: PRNGKeyArray) -> eqx.nn.MLP:
    """Replace every Linear weight in ``mlp`` with an orthogonally initialised tensor."""
    initializer = jax.nn.initializers.orthogonal()
    linears = [layer for layer in mlp.layers if isinstance(layer, eqx.nn.Linear)]
    keys = jax.random.split(key, len(linears))
    new_weights = [initializer(k, layer.weight.shape, jnp.float32) for layer, k in zip(linears, keys, strict=True)]

    def where_fn(m: eqx.nn.MLP) -> list[Float[Array, "out in"]]:
        return [layer.weight for layer in m.layers if isinstance(layer, eqx.nn.Linear)]

    return eqx.tree_at(where_fn, mlp, new_weights)


class MLP(eqx.Module):
    """Multi-layer perceptron with ReLU activations and orthogonal weight initialisation."""

    net: eqx.nn.MLP

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        *,
        key: PRNGKeyArray,
    ):
        """Initialise an MLP and overwrite its weights with orthogonal samples."""
        init_key, ortho_key = jax.random.split(key)
        base = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            key=init_key,
        )
        self.net = _apply_orthogonal(base, ortho_key)

    def __call__(self, x: Float[Array, " d"]) -> Float[Array, " out"]:
        """Forward pass through the MLP."""
        return self.net(x)
