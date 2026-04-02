import pytest
import torch.optim

from rltrain.utils.builders.load import load


def test_load_torch_adam():
    cls = load("torch.optim.Adam")
    assert cls is torch.optim.Adam


def test_load_torch_relu():
    cls = load("torch.nn.ReLU")
    assert cls is torch.nn.ReLU


def test_load_invalid_module_raises():
    with pytest.raises(ModuleNotFoundError):
        load("nonexistent.module.Class")


def test_load_invalid_attr_raises():
    with pytest.raises(AttributeError):
        load("torch.optim.NonExistentOptimizer")
