"""Tests for MLP network architectures."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from rltrain.networks import (
    D2RLMLP,
    MLP,
    ConvD2RLMLP,
    ConvFourierD2RLMLP,
    FourierBottleneck,
    fourier_feature_effective_rank,
    fourier_feature_sign_entropy,
)


@pytest.fixture
def key():
    return jax.random.key(0)


@pytest.mark.unit
def test_mlp_forward_shape(key):
    """Given an 8-d observation, MLP produces an out-sized vector."""
    # Given
    net = MLP(in_size=8, out_size=4, width_size=32, depth=2, key=key)
    obs = jnp.zeros(8)

    # When
    y = net(obs)

    # Then
    assert y.shape == (4,)


@pytest.mark.unit
def test_d2rl_mlp_forward_shape(key):
    """Given an 8-d observation, D2RLMLP produces an out-sized vector."""
    # Given
    net = D2RLMLP(in_size=8, out_size=4, width_size=32, depth=3, key=key)
    obs = jnp.zeros(8)

    # When
    y = net(obs)

    # Then
    assert y.shape == (4,)


@pytest.mark.unit
def test_d2rl_mlp_param_counts_match_dense_skip_topology(key):
    """Given depth=L, the dense-skip topology shapes the hidden weight matrices correctly."""
    # Given
    in_size, out_size, width, depth = 8, 4, 32, 3

    # When
    net = D2RLMLP(in_size=in_size, out_size=out_size, width_size=width, depth=depth, key=key)

    # Then: hidden_layers[0] takes raw obs (in_size -> width); subsequent layers take [h_prev; obs]
    assert net.hidden_layers[0].weight.shape == (width, in_size)
    for layer in net.hidden_layers[1:]:
        assert layer.weight.shape == (width, width + in_size)
    assert net.output_layer.weight.shape == (out_size, width)
    assert len(net.hidden_layers) == depth


@pytest.mark.unit
def test_d2rl_mlp_orthogonal_weights(key):
    """Given orthogonal initialisation, every linear weight satisfies W W^T ~ I (up to rank)."""
    # Given
    net = D2RLMLP(in_size=8, out_size=4, width_size=16, depth=2, key=key)

    # When / Then: orthogonal init means W @ W.T == I when rows are independent.
    # Includes the output projection — it goes through the same _orthogonal_linear helper.
    for layer in [*net.hidden_layers, net.output_layer]:
        w = layer.weight
        m = min(w.shape)
        # The smaller dimension's gram matrix should be the identity.
        gram = w @ w.T if w.shape[0] <= w.shape[1] else w.T @ w
        assert jnp.allclose(gram[:m, :m], jnp.eye(m), atol=1e-5)


@pytest.mark.unit
def test_d2rl_mlp_call_uses_dense_skip_concat(key):
    """Given a D2RLMLP, ablating the obs-slice columns of hidden layers >= 2
    must STRICTLY reduce __call__'s sensitivity to obs changes. A __call__ that
    silently dropped the concat would leave delta unchanged."""
    # Given
    width_size = 16
    net = D2RLMLP(in_size=8, out_size=4, width_size=width_size, depth=4, key=key)
    obs_a = jnp.zeros(8)
    obs_b = jnp.ones(8)

    # When: measure obs-sensitivity of the real network via __call__.
    delta_full = jnp.linalg.norm(net(obs_b) - net(obs_a))

    # And: ablate obs-slice columns of every hidden layer past the first
    # (these are the columns dense-skip routes obs through; layer 0 has no concat).
    def zero_obs_slice(layer):
        ablated_w = layer.weight.at[:, width_size:].set(0.0)
        return eqx.tree_at(lambda lin: lin.weight, layer, ablated_w)

    ablated_hidden = (
        net.hidden_layers[0],
        *(zero_obs_slice(layer) for layer in net.hidden_layers[1:]),
    )
    net_ablated = eqx.tree_at(lambda n: n.hidden_layers, net, ablated_hidden)
    delta_ablated = jnp.linalg.norm(net_ablated(obs_b) - net_ablated(obs_a))

    # Then: ablating must strictly reduce sensitivity. Otherwise __call__ wasn't
    # using the concat in the first place.
    assert delta_ablated < delta_full, (
        f"Ablating obs-slice columns did not reduce obs-sensitivity "
        f"(full={delta_full:.4f}, ablated={delta_ablated:.4f}) — "
        f"__call__ probably bypasses the dense-skip concat."
    )


@pytest.mark.unit
def test_d2rl_mlp_rejects_invalid_depth(key):
    """Given depth=0, D2RLMLP raises ValueError."""
    # Given / When / Then
    with pytest.raises(ValueError, match="depth >= 1"):
        D2RLMLP(in_size=8, out_size=4, width_size=16, depth=0, key=key)


@pytest.mark.unit
def test_d2rl_mlp_is_eqx_module(key):
    """Given a constructed D2RLMLP, it is an Equinox module suitable for jit/vmap."""
    # Given
    net = D2RLMLP(in_size=8, out_size=4, width_size=16, depth=2, key=key)

    # When / Then
    assert isinstance(net, eqx.Module)
    # vmap over a batch of obs should work
    batched = jax.vmap(net)(jnp.zeros((5, 8)))
    assert batched.shape == (5, 4)


@pytest.mark.unit
def test_conv_d2rl_mlp_forward_shape(key):
    """Given a (10, 10, 4) image obs, ConvD2RLMLP produces an out-sized vector
    and vmap'd batches preserve the leading axis."""
    # Given
    net = ConvD2RLMLP(
        height=10,
        width=10,
        in_channels=4,
        out_size=5,
        conv_channels=16,
        conv_kernel=3,
        feature_dim=128,
        mlp_width=256,
        mlp_depth=4,
        key=key,
    )
    obs = jnp.zeros((10, 10, 4))

    # When
    y = net(obs)
    batched = jax.vmap(net)(jnp.zeros((7, 10, 10, 4)))

    # Then
    assert y.shape == (5,)
    assert batched.shape == (7, 5)


@pytest.mark.unit
def test_conv_d2rl_mlp_param_count_topology(key):
    """Given default Breakout-MinAtar dims, the conv, projection, and D2RL
    backbone weights satisfy the principled topology: the projection actually
    projects from the conv flat-dim down to feature_dim, and the D2RL backbone
    sees feature_dim (not the conv flat-dim) on its first hidden layer."""
    # Given
    net = ConvD2RLMLP(
        height=10,
        width=10,
        in_channels=4,
        out_size=3,
        conv_channels=16,
        conv_kernel=3,
        feature_dim=128,
        mlp_width=256,
        mlp_depth=4,
        key=key,
    )

    # When / Then: conv weight is (C_out, C_in, k, k)
    assert net.conv.weight.shape == (16, 4, 3, 3)

    # Projection takes the flattened conv output (16 * 8 * 8 = 1024) to feature_dim.
    # H' = 10 - 3 + 1 = 8, W' = 8.
    assert net.projection.weight.shape == (128, 1024)

    # D2RL backbone's first hidden layer takes feature_dim (128) not flat_dim (1024).
    # Hidden weight shape is (width, in_size) for the first layer.
    assert net.d2rl.hidden_layers[0].weight.shape == (256, 128)
    # And subsequent hidden layers use the [h_prev; z] concat: (256, 256 + 128).
    for layer in net.d2rl.hidden_layers[1:]:
        assert layer.weight.shape == (256, 256 + 128)
    assert net.d2rl.output_layer.weight.shape == (3, 256)


@pytest.mark.unit
def test_conv_d2rl_mlp_orthogonal_weights(key):
    """Given orthogonal init, conv, projection, and every D2RL Linear satisfy
    W W^T ~ I on the smaller axis."""
    # Given
    net = ConvD2RLMLP(
        height=10,
        width=10,
        in_channels=4,
        out_size=3,
        conv_channels=16,
        conv_kernel=3,
        feature_dim=128,
        mlp_width=256,
        mlp_depth=4,
        key=key,
    )

    def gram_is_identity(w):
        m = min(w.shape)
        gram = w @ w.T if w.shape[0] <= w.shape[1] else w.T @ w
        return jnp.allclose(gram[:m, :m], jnp.eye(m), atol=1e-5)

    # When / Then: reshape conv weight (C_out, C_in, k, k) -> (C_out, C_in * k * k)
    conv_w = net.conv.weight
    conv_2d = conv_w.reshape(conv_w.shape[0], -1)
    assert gram_is_identity(conv_2d), "conv weight is not orthogonal"
    assert gram_is_identity(net.projection.weight), "projection weight is not orthogonal"
    for layer in [*net.d2rl.hidden_layers, net.d2rl.output_layer]:
        assert gram_is_identity(layer.weight), "D2RL backbone Linear is not orthogonal"


@pytest.mark.unit
def test_conv_d2rl_mlp_obs_reaches_d2rl_backbone(key):
    """Given two different observations, the network produces different outputs.
    The conv + projection + ReLU chain is not cleanly ablatable (ReLU can zero
    out signal even when topology is correct), so this is a light end-to-end
    behavioural probe that the obs influences the output."""
    # Given
    net = ConvD2RLMLP(
        height=10,
        width=10,
        in_channels=4,
        out_size=3,
        conv_channels=16,
        conv_kernel=3,
        feature_dim=128,
        mlp_width=256,
        mlp_depth=4,
        key=key,
    )
    obs_a = jnp.zeros((10, 10, 4))
    # Non-zero observation: full of ones so every channel and spatial location is excited.
    obs_b = jnp.ones((10, 10, 4))

    # When
    y_a = net(obs_a)
    y_b = net(obs_b)

    # Then
    assert not jnp.allclose(y_a, y_b), (
        f"Outputs for distinct observations are equal: {y_a} == {y_b} — the obs is not reaching the D2RL backbone."
    )


# ----- FourierBottleneck -----------------------------------------------------


@pytest.mark.unit
def test_fourier_bottleneck_forward_shape(key):
    """Given a (1024,) input, FourierBottleneck produces a (128,) output."""
    # Given
    layer = FourierBottleneck(in_dim=1024, out_dim=128, n_freqs=256, key=key)
    x = jnp.zeros(1024)

    # When
    y = layer(x)

    # Then
    assert y.shape == (128,)


@pytest.mark.unit
def test_fourier_bottleneck_param_shapes(key):
    """Verify B is (n_freqs, in_dim) and weight is (out_dim, 2*n_freqs).

    The 2*n_freqs factor confirms the sin/cos pair widens the post-Fourier
    feature dim before the learned projection — a swapped n_freqs/2*n_freqs
    would be silent without this check.
    """
    # Given
    layer = FourierBottleneck(in_dim=1024, out_dim=128, n_freqs=64, key=key)

    # Then
    assert layer.B.shape == (64, 1024), f"B shape mismatch: {layer.B.shape}"
    assert layer.weight.shape == (128, 128), f"weight shape mismatch: {layer.weight.shape}"
    assert layer.bias.shape == (128,)


@pytest.mark.unit
def test_fourier_bottleneck_frequency_matrix_is_frozen(key):
    """Given a scalar loss on the layer's output, the gradient w.r.t. B is exactly zero.

    This is the load-bearing claim — if stop_gradient ever gets dropped, the
    layer silently becomes "learned Fourier features" and the plasticity-
    preservation rationale collapses.
    """
    # Given
    layer = FourierBottleneck(in_dim=64, out_dim=8, n_freqs=16, key=key)
    x = jax.random.normal(jax.random.key(1), (4, 64))

    # When we differentiate a mean-squared output through the layer.
    def loss_fn(m, batch):
        return jnp.mean(jax.vmap(m)(batch) ** 2)

    grads = eqx.filter_grad(loss_fn)(layer, x)

    # Then: gradient on B is exactly zero (stop_gradient cuts the path).
    # Gradients on weight/bias must be nonzero (otherwise the layer doesn't learn at all).
    assert jnp.all(grads.B == 0.0), f"B received non-zero gradient (max |g|={float(jnp.max(jnp.abs(grads.B))):.2e})"
    assert float(jnp.max(jnp.abs(grads.weight))) > 0.0, "weight got zero gradient — layer can't learn"


@pytest.mark.unit
def test_fourier_bottleneck_vmap_batched(key):
    """The layer composes cleanly under jax.vmap over a batch dim."""
    # Given
    layer = FourierBottleneck(in_dim=32, out_dim=16, n_freqs=8, key=key)
    batch = jnp.zeros((5, 32))

    # When
    y = jax.vmap(layer)(batch)

    # Then
    assert y.shape == (5, 16)


@pytest.mark.unit
def test_fourier_feature_effective_rank_full_for_random(key):
    """Random unit-variance features have near-maximal effective rank.

    Sanity: the probe should treat orthogonal/full-rank features as "diverse"
    (effective rank close to feature dim, not collapsed to 1).
    """
    # Given a batch of random features with no internal collapse.
    feats = jax.random.normal(key, (64, 32))

    # When
    rank = fourier_feature_effective_rank(feats)

    # Then: effective rank should be a meaningful fraction of feature dim.
    # With 64 samples and 32 features, full-rank random gives rank > 20.
    assert float(rank) > 20.0, f"random features should have rank >>1, got {float(rank):.2f}"


@pytest.mark.unit
def test_fourier_feature_effective_rank_collapses_for_rank_one(key):
    """Rank-1 features (every row equals the same vector) collapse to rank ~= 1."""
    # Given a batch of identical rows — rank-1 by construction.
    v = jax.random.normal(key, (32,))
    feats = jnp.broadcast_to(v, (64, 32))

    # When
    rank = fourier_feature_effective_rank(feats)

    # Then: after centring, the matrix is exactly zero; the probe should report
    # effective rank close to 1 (exp(0) when entropy is over a degenerate
    # spectrum). Concretely: with eps=1e-12 and all singular values ~0,
    # all probabilities normalise to ~1/n giving exp(log(n)) = n — but the
    # *centred* matrix is rank zero, so this tests the "no spread" floor.
    # We just check it's NOT the random-features high-rank case.
    assert float(rank) < 5.0, f"degenerate features should collapse to low rank, got {float(rank):.2f}"


@pytest.mark.unit
def test_fourier_feature_sign_entropy_full_for_balanced(key):
    """Symmetric-around-zero features have near-maximal mean sign entropy."""
    # Given a balanced (mean 0, symmetric) batch of features.
    feats = jax.random.normal(key, (64, 32))

    # When
    ent = fourier_feature_sign_entropy(feats)

    # Then: with ~50/50 sign split per unit, mean entropy is close to 1.
    assert 0.85 < float(ent) <= 1.0, f"balanced features should have sign entropy ~1, got {float(ent):.3f}"


@pytest.mark.unit
def test_fourier_feature_sign_entropy_zero_for_constant_sign(key):
    """All-positive features have sign entropy ~= 0 (saturated)."""
    # Given features that are always positive (e.g. ReLU'd random or exp'd).
    feats = jnp.abs(jax.random.normal(key, (64, 32))) + 0.1

    # When
    ent = fourier_feature_sign_entropy(feats)

    # Then: with all units positive on every input, entropy → 0.
    assert float(ent) < 0.05, f"constant-sign features should have entropy ~0, got {float(ent):.4f}"


# ----- ConvFourierD2RLMLP ----------------------------------------------------


@pytest.mark.unit
def test_conv_fourier_d2rl_mlp_forward_shape(key):
    """Given a (10,10,4) image, ConvFourierD2RLMLP produces an out-sized vector."""
    # Given
    net = ConvFourierD2RLMLP(
        height=10,
        width=10,
        in_channels=4,
        out_size=3,
        conv_channels=16,
        conv_kernel=3,
        feature_dim=128,
        n_freqs=64,  # smaller for test speed
        mlp_width=128,
        mlp_depth=2,
        key=key,
    )
    obs = jnp.zeros((10, 10, 4))

    # When
    y = net(obs)

    # Then
    assert y.shape == (3,)


@pytest.mark.unit
def test_conv_fourier_d2rl_mlp_bottleneck_is_fourier(key):
    """The bottleneck inside ConvFourierD2RLMLP is a FourierBottleneck instance
    (not a plain Linear). Distinguishes this class from ConvD2RLMLP at the
    structural level — accidental swap-back would silently regress the
    plasticity-preservation guarantee."""
    # Given
    net = ConvFourierD2RLMLP(height=10, width=10, in_channels=4, out_size=3, n_freqs=8, mlp_depth=2, key=key)

    # Then
    assert isinstance(net.bottleneck, FourierBottleneck)


@pytest.mark.unit
def test_conv_fourier_d2rl_mlp_obs_reaches_output(key):
    """Distinct observations produce distinct outputs (light end-to-end probe)."""
    # Given
    net = ConvFourierD2RLMLP(
        height=10,
        width=10,
        in_channels=4,
        out_size=3,
        conv_channels=16,
        conv_kernel=3,
        feature_dim=128,
        n_freqs=64,
        mlp_width=128,
        mlp_depth=2,
        key=key,
    )
    obs_a = jnp.zeros((10, 10, 4))
    obs_b = jnp.ones((10, 10, 4))

    # When
    y_a = net(obs_a)
    y_b = net(obs_b)

    # Then
    assert not jnp.allclose(y_a, y_b)
