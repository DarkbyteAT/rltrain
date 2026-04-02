"""Gradient transform pipeline for composable optimisation techniques.

A gradient transform is a step applied between ``loss.backward()`` and
``optimizer.step()`` inside ``Agent.learn()``.  Transforms can modify
gradients, temporarily perturb parameters, recompute the loss, or
accumulate state across steps.

The pipeline is configured via the ``grad_transforms`` key in agent JSON::

    "grad_transforms": [
        {"fqn": "rltrain.transforms.SAM", "rho": 1e-2},
        {"fqn": "rltrain.transforms.LAMPRollback", "eps": 5e-3, "rollback_len": 10}
    ]
"""

from rltrain.transforms.asam import ASAM as ASAM
from rltrain.transforms.lamp import LAMPRollback as LAMPRollback
from rltrain.transforms.protocol import GradientTransform as GradientTransform
from rltrain.transforms.sam import SAM as SAM
