from types import ModuleType
from typing import Callable, Union

def load(fqn: str) -> Union[ModuleType, type, Callable]:
    """Returns a module/class/function from the given fully-qualified name.
    
    Parameters
    ----------
    ``fqn``
        The fully-qualified name of the module/class/function to import.
    """
    
    parts = fqn.split(".")
    module = ".".join(parts[:-1])
    root = __import__(module)
    for sub in parts[1:]:
        root = getattr(root, sub)
    return root