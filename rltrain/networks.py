"""MLP networks with orthogonal weight initialisation."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


def _orthogonal_linear(in_features: int, out_features: int, key: PRNGKeyArray) -> eqx.nn.Linear:
    """Create an `eqx.nn.Linear` and overwrite its weight with an orthogonal sample."""
    w_key, b_key = jax.random.split(key)
    linear = eqx.nn.Linear(in_features, out_features, key=b_key)
    initializer = jax.nn.initializers.orthogonal()
    new_weight = initializer(w_key, linear.weight.shape, jnp.float32)
    return eqx.tree_at(lambda lin: lin.weight, linear, new_weight)


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


class D2RLMLP(eqx.Module):
    r"""Dense-residual MLP from D2RL (Sinha et al. 2020).

    Each hidden layer past the first concatenates the previous hidden activation
    with the raw observation before its linear projection, giving every hidden
    layer direct access to the input. This dense-skip topology improves training
    of deep RL value and policy networks without changing the optimiser, loss,
    or algorithm.

    Layer-wise, with input $x \in \mathbb{R}^{d}$ and hidden width $h$:

    $$
    \begin{aligned}
        z_1 &= \mathrm{ReLU}(W_1 x + b_1), & W_1 &\in \mathbb{R}^{h \times d} \\
        z_i &= \mathrm{ReLU}(W_i [z_{i-1}; x] + b_i), & W_i &\in \mathbb{R}^{h \times (h + d)} \quad (i = 2,\dots,L) \\
        y   &= W_\mathrm{out} z_L + b_\mathrm{out}, & W_\mathrm{out} &\in \mathbb{R}^{o \times h}
    \end{aligned}
    $$

    The output projection consumes only the last hidden activation — no final
    observation concat — matching the canonical D2RL formulation.

    References:
        Sinha, S., Bharadhwaj, H., Srinivas, A., & Garg, A. (2020).
        D2RL: Deep Dense Architectures in Reinforcement Learning.
        arXiv:2010.09163.
    """

    hidden_layers: tuple[eqx.nn.Linear, ...]
    output_layer: eqx.nn.Linear
    in_size: int = eqx.field(static=True)

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        *,
        key: PRNGKeyArray,
    ):
        """Initialise a D2RL MLP with orthogonal weights on every linear layer.

        Args:
            in_size: Dimension of the input observation.
            out_size: Dimension of the output (feature dim for actors, 1 for critics).
            width_size: Hidden width $h$ shared across every hidden layer.
            depth: Number of hidden layers $L$. Must be at least 1.
            key: PRNG key used to seed orthogonal initialisation.

        Raises:
            ValueError: If ``depth`` is less than 1.
        """
        if depth < 1:
            raise ValueError(f"D2RLMLP requires depth >= 1, got {depth}")

        keys = jax.random.split(key, depth + 1)
        hidden: list[eqx.nn.Linear] = []
        for i in range(depth):
            layer_in = in_size if i == 0 else width_size + in_size
            hidden.append(_orthogonal_linear(layer_in, width_size, keys[i]))
        self.hidden_layers = tuple(hidden)
        self.output_layer = _orthogonal_linear(width_size, out_size, keys[depth])
        self.in_size = in_size

    def __call__(self, x: Float[Array, " d"]) -> Float[Array, " out"]:
        """Forward pass through the dense-residual MLP."""
        h = jax.nn.relu(self.hidden_layers[0](x))
        for layer in self.hidden_layers[1:]:
            h = jax.nn.relu(layer(jnp.concatenate([h, x], axis=-1)))
        return self.output_layer(h)
