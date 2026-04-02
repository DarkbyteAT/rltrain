"""Device resolution for training backends.

rltrain is device-agnostic — the Agent constructor takes a ``T.device`` and all
tensor operations use ``.to(self.device)``. This module provides the auto-detection
logic that maps a user-facing device string to a concrete PyTorch device.

Supported backends:

- ``cpu`` — always available
- ``cuda`` — NVIDIA GPUs via CUDA
- ``mps`` — Apple Silicon GPUs via Metal Performance Shaders
- ``auto`` — selects the best available backend automatically
"""

import torch as T


def resolve_device(device: str) -> T.device:
    """Resolve a device string to a concrete ``torch.device``.

    Parameters
    ----------
    ``device`` : ``str``
        One of ``"cpu"``, ``"cuda"``, ``"mps"``, or ``"auto"``.
        ``"auto"`` selects the best available backend.

    Returns
    -------
    ``torch.device``
        The resolved device.

    Raises
    ------
    ``ValueError``
        If the requested device is not available.
    """
    if device == "auto":
        if T.cuda.is_available():
            return T.device("cuda")
        if T.backends.mps.is_available():
            return T.device("mps")
        return T.device("cpu")

    resolved = T.device(device)

    if device == "cuda" and not T.cuda.is_available():
        raise ValueError("CUDA requested but not available")
    if device == "mps" and not T.backends.mps.is_available():
        raise ValueError("MPS requested but not available")

    return resolved
