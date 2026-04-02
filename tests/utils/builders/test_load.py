from functools import partial

import pytest
import torch.nn
import torch.optim

from rltrain.utils.builders.load import load, resolve


# --- load -------------------------------------------------------------------


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


# --- resolve: plain values --------------------------------------------------


@pytest.mark.parametrize("value", [42, 3.14, "hello", True, None])
def test_resolve_plain_values_pass_through(value):
    """Scalars are returned unchanged."""
    # Given -- a plain value
    # When
    result = resolve(value)
    # Then
    assert result is value


# --- resolve: dict with fqn -------------------------------------------------


def test_resolve_dict_with_fqn_constructs_object():
    """A dict containing an 'fqn' key is resolved into a constructed object."""
    # Given
    cfg = {"fqn": "torch.nn.Linear", "in_features": 4, "out_features": 2}

    # When
    obj = resolve(cfg)

    # Then
    assert isinstance(obj, torch.nn.Linear)
    assert obj.in_features == 4
    assert obj.out_features == 2


# --- resolve: list of fqn dicts ---------------------------------------------


def test_resolve_list_of_fqn_dicts():
    """A list of fqn dicts resolves to a list of constructed objects."""
    # Given
    cfg = [
        {"fqn": "torch.nn.Linear", "in_features": 4, "out_features": 8},
        {"fqn": "torch.nn.ReLU"},
        {"fqn": "torch.nn.Linear", "in_features": 8, "out_features": 2},
    ]

    # When
    modules = resolve(cfg)

    # Then
    assert len(modules) == 3
    assert isinstance(modules[0], torch.nn.Linear)
    assert isinstance(modules[1], torch.nn.ReLU)
    assert isinstance(modules[2], torch.nn.Linear)


# --- resolve: nested fqn dicts ----------------------------------------------


def test_resolve_nested_fqn_in_kwargs():
    """An fqn dict whose kwargs contain another fqn dict resolves recursively."""
    # Given -- a dict without fqn whose values contain fqn dicts (nested resolution)
    cfg = {
        "policy": {"fqn": "torch.nn.Linear", "in_features": 4, "out_features": 2},
        "activation": {"fqn": "torch.nn.ReLU"},
    }

    # When
    result = resolve(cfg)

    # Then
    assert isinstance(result["policy"], torch.nn.Linear)
    assert isinstance(result["activation"], torch.nn.ReLU)


# --- resolve: deferred construction ------------------------------------------


def test_resolve_deferred_produces_partial():
    """A dict with 'deferred: true' wraps in functools.partial."""
    # Given
    cfg = {"fqn": "torch.optim.Adam", "lr": 3e-4, "deferred": True}

    # When
    factory = resolve(cfg)

    # Then
    assert isinstance(factory, partial)
    assert factory.func is torch.optim.Adam
    assert factory.keywords == {"lr": 3e-4}


def test_resolve_deferred_does_not_pass_deferred_key():
    """The 'deferred' key must not appear in the partial's kwargs."""
    # Given
    cfg = {"fqn": "torch.optim.SGD", "lr": 0.01, "deferred": True}

    # When
    factory = resolve(cfg)

    # Then
    assert "deferred" not in factory.keywords


# --- resolve: dict without fqn recurses into values -------------------------


def test_resolve_dict_without_fqn_recurses():
    """A dict without 'fqn' has its values recursively resolved."""
    # Given
    cfg = {
        "layer": {"fqn": "torch.nn.Linear", "in_features": 4, "out_features": 2},
        "name": "my_layer",
        "count": 3,
    }

    # When
    result = resolve(cfg)

    # Then
    assert isinstance(result["layer"], torch.nn.Linear)
    assert result["name"] == "my_layer"
    assert result["count"] == 3


# --- resolve: edge cases ------------------------------------------------


def test_resolve_empty_dict():
    """An empty dict resolves to an empty dict."""
    assert resolve({}) == {}


def test_resolve_empty_list():
    """An empty list resolves to an empty list."""
    assert resolve([]) == []


# --- resolve: error propagation ------------------------------------------


def test_resolve_invalid_fqn_raises_with_context():
    """resolve() wraps FQN load errors with the failing fqn for debugging."""
    # Given
    cfg = {"fqn": "nonexistent.module.Class", "x": 1}

    # When / Then
    with pytest.raises(ModuleNotFoundError, match="nonexistent.module.Class"):
        resolve(cfg)


def test_resolve_invalid_attr_raises_with_context():
    """resolve() wraps attribute errors with the failing fqn for debugging."""
    # Given
    cfg = {"fqn": "torch.optim.NonExistentOptimizer"}

    # When / Then
    with pytest.raises(AttributeError, match="torch.optim.NonExistentOptimizer"):
        resolve(cfg)
