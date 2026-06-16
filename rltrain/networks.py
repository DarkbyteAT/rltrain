"""MLP networks with orthogonal weight initialisation."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


def _orthogonal_linear(in_features: int, out_features: int, key: PRNGKeyArray) -> eqx.nn.Linear:
    """Create an `eqx.nn.Linear` and overwrite its weight with an orthogonal sample."""
    w_key, linear_key = jax.random.split(key)
    # ``linear_key`` seeds eqx.nn.Linear's default (LeCun-uniform) init for both weight
    # and bias; the weight is then immediately overwritten with the orthogonal sample,
    # so only the bias init survives.
    linear = eqx.nn.Linear(in_features, out_features, key=linear_key)
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


class ConvD2RLMLP(eqx.Module):
    r"""Convolutional feature extractor feeding a D2RL dense-residual MLP.

    Adapts D2RL (Sinha et al. 2020) to image-shaped observations. The original
    formulation concatenates the raw observation with each hidden activation
    inside the MLP — sensible for low-dimensional vector inputs but not for
    raw images, where the dense skip would dominate the hidden activation and
    blow up parameter counts.

    The principled extension here is to insert a small learned projection
    *before* D2RL takes over:

    1. A single ``Conv2d`` extracts spatial features from the image.
    2. A ``Linear`` projects the flattened conv output to ``feature_dim``.
    3. The ``feature_dim``-sized feature vector is fed to a standard
       :class:`D2RLMLP`. The dense-skip concat operates on this projected
       feature vector, matching the canonical vector-input shape that D2RL
       was designed for.

    The architecture, with image $x \in \mathbb{R}^{H \times W \times C}$:

    $$
    \begin{aligned}
        f &= \mathrm{ReLU}(\mathrm{Conv}(x_{CHW})) \in \mathbb{R}^{C' \times H' \times W'} \\
        z &= \mathrm{ReLU}(W_{\mathrm{proj}} \cdot \mathrm{vec}(f) + b_{\mathrm{proj}}) \in \mathbb{R}^{d} \\
        y &= \mathrm{D2RLMLP}(z)
    \end{aligned}
    $$

    where $d$ is ``feature_dim``, $H' = H - k + 1$ and $W' = W - k + 1$ for
    kernel size $k$ with stride 1 and no padding.

    Observations from gymnax MinAtar envs arrive in ``(H, W, C)`` (HWC)
    layout; ``__call__`` transposes them to ``(C, H, W)`` (CHW) before the
    conv, matching ``eqx.nn.Conv2d``'s expected layout.

    References:
        Sinha, S., Bharadhwaj, H., Srinivas, A., & Garg, A. (2020).
        D2RL: Deep Dense Architectures in Reinforcement Learning.
        arXiv:2010.09163.
    """

    conv: eqx.nn.Conv2d
    projection: eqx.nn.Linear
    d2rl: D2RLMLP
    height: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)

    def __init__(
        self,
        height: int,
        width: int,
        in_channels: int,
        out_size: int,
        *,
        conv_channels: int = 16,
        conv_kernel: int = 3,
        feature_dim: int = 128,
        mlp_width: int = 256,
        mlp_depth: int = 4,
        key: PRNGKeyArray,
    ):
        r"""Initialise a Conv -> ReLU -> Linear -> ReLU -> D2RLMLP stack.

        Args:
            height: Input image height $H$.
            width: Input image width $W$.
            in_channels: Number of input channels $C$ in the HWC observation.
            out_size: Dimension of the final output (action features for actors,
                $|\mathcal{A}|$ for discrete Q-critics).
            conv_channels: Number of conv output channels $C'$.
            conv_kernel: Conv kernel size $k$ (square, stride 1, no padding).
            feature_dim: Projected feature dimension $d$ fed into the D2RL MLP.
            mlp_width: Hidden width inside the D2RL MLP.
            mlp_depth: Depth (number of hidden layers) of the D2RL MLP.
            key: PRNG key seeding orthogonal init for conv, projection, and MLP.
        """
        conv_key, proj_key, mlp_key = jax.random.split(key, 3)

        # Conv: build with default init, then overwrite weight orthogonally.
        # eqx.nn.Conv2d's default conv_key seeds the bias init that survives.
        conv = eqx.nn.Conv2d(
            in_channels=in_channels,
            out_channels=conv_channels,
            kernel_size=conv_kernel,
            stride=1,
            padding=0,
            key=conv_key,
        )
        # Orthogonal init: sample a 2D orthogonal matrix shaped
        # ``(out_channels, in_channels * k * k)`` and reshape to the conv's
        # 4D weight. The default ``jax.nn.initializers.orthogonal()`` only
        # guarantees orthogonality on a 2D view (its ``column_axis`` semantics
        # don't give a clean per-output-channel orthogonal frame on 4D shapes),
        # so we orthogonalise the flat matrix first and reshape.
        ortho_init = jax.nn.initializers.orthogonal()
        flat_in = in_channels * conv_kernel * conv_kernel
        ortho_mat = ortho_init(conv_key, (conv_channels, flat_in), jnp.float32)
        new_conv_weight = ortho_mat.reshape(conv_channels, in_channels, conv_kernel, conv_kernel)
        self.conv = eqx.tree_at(lambda c: c.weight, conv, new_conv_weight)

        conv_out_h = height - conv_kernel + 1
        conv_out_w = width - conv_kernel + 1
        flat_dim = conv_channels * conv_out_h * conv_out_w
        self.projection = _orthogonal_linear(flat_dim, feature_dim, proj_key)

        self.d2rl = D2RLMLP(
            in_size=feature_dim,
            out_size=out_size,
            width_size=mlp_width,
            depth=mlp_depth,
            key=mlp_key,
        )

        self.height = height
        self.width = width
        self.in_channels = in_channels

    def __call__(self, x: Float[Array, "H W C"]) -> Float[Array, " out"]:
        """Forward pass: HWC->CHW transpose, conv, project, D2RL MLP."""
        x_chw = jnp.transpose(x, (2, 0, 1))
        f = jax.nn.relu(self.conv(x_chw))
        z = jax.nn.relu(self.projection(f.reshape(-1)))
        return self.d2rl(z)
