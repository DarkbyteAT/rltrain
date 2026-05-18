"""Trainer package -- training loop with strategy dispatch.

Re-exports the public API so consumers write::

    from rltrain.trainer import Trainer, PythonLoop, ScanLoop
"""

from rltrain.trainer._carry import StepOutput, TrainCarry, TrainConfig
from rltrain.trainer._loops import PmapLoop, PythonLoop, ScanLoop, TrainingLoop
from rltrain.trainer._trainer import Trainer


__all__ = [
    "Trainer",
    "TrainCarry",
    "TrainConfig",
    "StepOutput",
    "TrainingLoop",
    "PythonLoop",
    "ScanLoop",
    "PmapLoop",
]
