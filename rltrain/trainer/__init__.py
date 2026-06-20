"""Trainer package -- training loop with strategy dispatch.

Re-exports the public API so consumers write::

    from rltrain.trainer import Trainer, MultiSeedTrainer, PythonLoop, ScanLoop
"""

from rltrain.trainer._carry import StepOutput, TrainCarry, TrainConfig
from rltrain.trainer._loops import MultiSeedScanLoop, PmapLoop, PythonLoop, ScanLoop, TrainingLoop
from rltrain.trainer._multi_seed_trainer import MultiSeedTrainer
from rltrain.trainer._trainer import Trainer


__all__ = [
    "Trainer",
    "MultiSeedTrainer",
    "TrainCarry",
    "TrainConfig",
    "StepOutput",
    "TrainingLoop",
    "PythonLoop",
    "ScanLoop",
    "PmapLoop",
    "MultiSeedScanLoop",
]
