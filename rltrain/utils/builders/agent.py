import torch as T
import torch.nn as nn

from rltrain.agents import Agent
from rltrain.utils.builders.load import load, resolve


def agent(
    fqn: str,
    model: dict[str, list[dict]],
    opt: dict[str, dict],
    device: T.device,
    **kwargs,
) -> Agent:
    """Build an `Agent` from a JSON config.

    Parameters
    ----------
    `fqn`
        Fully-qualified name of the agent class.
    `model`
        Mapping of network name to a list of module configs (each with an
        ``"fqn"`` key plus constructor kwargs).
    `opt`
        Mapping of optimiser name to an optimiser config (``"fqn"`` key
        plus constructor kwargs). Automatically resolved as deferred
        (``functools.partial``) since optimisers need ``model.parameters()``
        at ``setup()`` time, not at build time.
    `device`
        Torch device to place the agent on.
    `**kwargs`
        Extra keyword arguments forwarded to the agent constructor.
    """
    agent_type = load(fqn)

    _model = nn.ModuleDict({name: nn.Sequential(*resolve(modules)) for name, modules in model.items()})

    _opt = {name: resolve({**cfg, "deferred": True}) for name, cfg in opt.items()}

    return agent_type(model=_model, opt=_opt, device=device, **kwargs)
