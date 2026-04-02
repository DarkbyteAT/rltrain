import torch.nn as nn

from rltrain.utils.builders.load import resolve


def test_distinct_optimizer_configs_produce_distinct_optimizers():
    """Each optimizer factory must use its own config, not the last-defined one.

    Regression test for the closure-over-loop-variable bug where a lambda
    captured `_kwargs` by reference, causing all factories to silently share
    the final iteration's hyperparameters.
    """
    # Given -- two optimizer configs with different learning rates
    opt_config = {
        "actor": {"fqn": "torch.optim.Adam", "lr": 0.001, "deferred": True},
        "critic": {"fqn": "torch.optim.Adam", "lr": 0.0003, "deferred": True},
    }

    # When -- resolve produces partial factories
    factories = {name: resolve(cfg) for name, cfg in opt_config.items()}

    # Then -- each factory uses its own learning rate
    dummy = nn.Linear(4, 2)
    actor_opt = factories["actor"](params=dummy.parameters())
    critic_opt = factories["critic"](params=dummy.parameters())

    actor_lr = actor_opt.param_groups[0]["lr"]
    critic_lr = critic_opt.param_groups[0]["lr"]

    assert actor_lr == 0.001, f"actor lr should be 0.001, got {actor_lr}"
    assert critic_lr == 0.0003, f"critic lr should be 0.0003, got {critic_lr}"
    assert actor_lr != critic_lr, "actor and critic should have different learning rates"
