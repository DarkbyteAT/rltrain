from functools import partial

import torch.nn as nn

from rltrain.utils.builders.load import load


def _build_opt_factories(opt_config: dict[str, dict]) -> dict[str, partial]:
    """Reproduce the optimizer factory-building logic from agent.py."""
    factories = {}
    for name, kwargs in opt_config.items():
        opt_type = load(kwargs["fqn"])
        opt_kwargs = {k: v for k, v in kwargs.items() if k != "fqn"}
        factories[name] = partial(opt_type, **opt_kwargs)
    return factories


def test_distinct_optimizer_configs_produce_distinct_optimizers():
    """Each optimizer factory must use its own config, not the last-defined one.

    Regression test for the closure-over-loop-variable bug where a lambda
    captured `_kwargs` by reference, causing all factories to silently share
    the final iteration's hyperparameters.
    """
    opt_config = {
        "actor": {"fqn": "torch.optim.Adam", "lr": 0.001},
        "critic": {"fqn": "torch.optim.Adam", "lr": 0.0003},
    }

    factories = _build_opt_factories(opt_config)

    dummy = nn.Linear(4, 2)
    actor_opt = factories["actor"](params=dummy.parameters())
    critic_opt = factories["critic"](params=dummy.parameters())

    actor_lr = actor_opt.param_groups[0]["lr"]
    critic_lr = critic_opt.param_groups[0]["lr"]

    assert actor_lr == 0.001, f"actor lr should be 0.001, got {actor_lr}"
    assert critic_lr == 0.0003, f"critic lr should be 0.0003, got {critic_lr}"
    assert actor_lr != critic_lr, "actor and critic should have different learning rates"
