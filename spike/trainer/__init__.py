"""Trainer package -- training loop with strategy dispatch.

Re-exports the public API so consumers write::

    from spike.trainer import Trainer, PythonLoop, ScanLoop
"""

from spike.trainer._carry import StepOutput, TrainCarry, TrainConfig
from spike.trainer._loops import PmapLoop, PythonLoop, ScanLoop, TrainingLoop
from spike.trainer._trainer import Trainer


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
