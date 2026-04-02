"""Verify that network modules produce correct output shapes."""

import torch as T

from rltrain.nn.d2rl import SkipMLP
from rltrain.nn.mlp import mlp
from rltrain.nn.rff import RFF


def test_mlp_output_shape():
    net = mlp(features=(8, 64, 32, 4))
    x = T.randn(16, 8)
    out = net(x)
    assert out.shape == (16, 4)


def test_mlp_single_layer():
    net = mlp(features=(4, 2))
    x = T.randn(8, 4)
    out = net(x)
    assert out.shape == (8, 2)


def test_skip_mlp_output_shape():
    net = SkipMLP(inputs=4, hiddens=[32, 32], outputs=2)
    x = T.randn(16, 4)
    out = net(x)
    assert out.shape == (16, 2)


def test_skip_mlp_deep():
    net = SkipMLP(inputs=8, hiddens=[64, 64, 64, 64], outputs=3)
    x = T.randn(4, 8)
    out = net(x)
    assert out.shape == (4, 3)


def test_rff_output_shape():
    net = RFF(in_features=4, out_features=16)
    x = T.randn(8, 4)
    out = net(x)
    assert out.shape == (8, 16)


def test_rff_weights_are_frozen():
    net = RFF(in_features=4, out_features=8)
    for p in net.parameters():
        assert not p.requires_grad


def test_rff_rejects_odd_output():
    import pytest

    with pytest.raises(AssertionError):
        RFF(in_features=4, out_features=7)
