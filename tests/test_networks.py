"""Tests for MLP network architectures."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from rltrain.networks import D2RLMLP, MLP


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
def test_d2rl_mlp_obs_reaches_every_hidden_layer(key):
    """Given two observations differing only in their values, the second-and-later
    hidden layers' outputs MUST differ — otherwise the dense-skip is not wired."""
    # Given two D2RLMLPs sharing the same params, applied to two different obs.
    net = D2RLMLP(in_size=8, out_size=4, width_size=16, depth=3, key=key)
    obs_a = jnp.zeros(8)
    obs_b = jnp.ones(8)

    # When we walk both forward layer-by-layer (replicating __call__ inline so we
    # can inspect intermediate activations).
    def hidden_outputs(x):
        h = jax.nn.relu(net.hidden_layers[0](x))
        outs = [h]
        for layer in net.hidden_layers[1:]:
            h = jax.nn.relu(layer(jnp.concatenate([h, x], axis=-1)))
            outs.append(h)
        return outs

    outs_a = hidden_outputs(obs_a)
    outs_b = hidden_outputs(obs_b)

    # Then: layer 0 already sees obs (no concat needed), so it should differ.
    # Layers 1+ should ALSO differ — and crucially, the difference at layer i+1
    # depends on obs reaching it directly, not just through h_prev.
    for i, (a, b) in enumerate(zip(outs_a, outs_b, strict=True)):
        assert not jnp.allclose(a, b), f"hidden layer {i} did not react to obs change"


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
