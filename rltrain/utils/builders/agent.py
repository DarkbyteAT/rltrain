from functools import partial

import torch as T
import torch.nn as nn

from rltrain.agents import Agent
from rltrain.utils.builders import load

def agent(fqn: str,
            model: dict[str, list[dict[str]]],
            opt: dict[str, dict[str]],
            device: T.device,
            **kwargs
    ) -> Agent:
    
    # Gets the class of the agent being created
    agent_type = load(fqn)
    _model = nn.ModuleDict()
    _opt = {}
    
    # Iterates over all networks defined
    for name, _kwargs in model.items():
        modules = []
        
        # Iterates over all modules in the network and creates them
        for _module in _kwargs:
            module_type = load(_module["fqn"])
            module = module_type(**{k : v for k, v in _module.items() if k != "fqn"})
            modules.append(module)
        
        # Adds the created module to the network
        _model[name] = nn.Sequential(*modules)
    
    # Iterates over all optimisers defined to create builder functions
    for name, _kwargs in opt.items():
        opt_type = load(_kwargs["fqn"])
        opt_kwargs = {k: v for k, v in _kwargs.items() if k != "fqn"}
        _opt[name] = partial(opt_type, **opt_kwargs)
    
    return agent_type(model=_model, opt=_opt, device=device, **kwargs)