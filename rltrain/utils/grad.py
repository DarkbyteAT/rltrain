import torch as T
import torch.nn as nn

from typing import Iterable

def get_grad(params: Iterable[nn.Parameter]) -> T.Tensor:
    """Returns the current gradient of each parameter as a flattened vector.
    
    Parameters
    ----------
    ``params`` : ``Iterable[Parameter]``
        An ``Iterable`` of the parameters to collect gradients from.
    
    Returns
    -------
    ``Tensor``
        A flattened vector of the gradients of each parameter.
    """
    
    return T.cat([p.grad.view(-1) for p in params])

@T.no_grad()
def set_grad(grads: T.Tensor, params: Iterable[nn.Parameter]):
    """Sets the gradient of each parameter to the corresponding elements of a vector.
    
    Parameters
    ----------
    ``grads`` : ``Tensor``
        A flattened vector of the intended gradient for each parameter.
    ``params`` : ``Iterable[Parameter]``
        An ``Iterable`` of parameters to update the gradients of.
    """
    
    i = 0
    
    for p in params:
        p.grad = grads[i:i+p.numel()].view(p.shape)
        i += p.numel()