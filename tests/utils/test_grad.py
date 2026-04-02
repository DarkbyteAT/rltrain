import torch as T
import torch.nn as nn

from rltrain.utils.grad import get_grad, set_grad


def test_get_grad_shape():
    layer = nn.Linear(4, 2, bias=True)
    x = T.randn(1, 4)
    loss = layer(x).sum()
    loss.backward()

    params = list(layer.parameters())
    grads = get_grad(params)
    total_params = sum(p.numel() for p in params)
    assert grads.shape == (total_params,)


def test_set_grad_overwrites():
    layer = nn.Linear(3, 2, bias=False)
    x = T.randn(1, 3)
    loss = layer(x).sum()
    loss.backward()

    params = list(layer.parameters())
    total_params = sum(p.numel() for p in params)
    new_grads = T.ones(total_params)
    set_grad(new_grads, params)

    for p in params:
        assert T.allclose(p.grad, T.ones_like(p.grad))


def test_get_set_roundtrip():
    layer = nn.Linear(4, 3, bias=True)
    x = T.randn(1, 4)
    loss = layer(x).sum()
    loss.backward()

    params = list(layer.parameters())
    original_grads = get_grad(params).clone()

    # Zero grads, set them back, and verify
    for p in params:
        p.grad = T.zeros_like(p.grad)

    set_grad(original_grads, params)
    restored = get_grad(params)
    assert T.allclose(original_grads, restored)
