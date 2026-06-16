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
