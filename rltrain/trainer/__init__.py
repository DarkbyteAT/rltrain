"""Trainer package -- training loop with strategy dispatch.

Re-exports the public API so consumers write::

    from rltrain.trainer import Trainer, MultiSeedTrainer, PythonLoop, ScanLoop
"""

from rltrain.trainer._carry import StepOutput as StepOutput
from rltrain.trainer._carry import TrainCarry as TrainCarry
from rltrain.trainer._carry import TrainConfig as TrainConfig
from rltrain.trainer._loops import MultiSeedScanLoop as MultiSeedScanLoop
from rltrain.trainer._loops import PmapLoop as PmapLoop
from rltrain.trainer._loops import PythonLoop as PythonLoop
from rltrain.trainer._loops import ScanLoop as ScanLoop
from rltrain.trainer._loops import TrainingLoop as TrainingLoop
from rltrain.trainer._multi_seed_trainer import MultiSeedTrainer as MultiSeedTrainer
from rltrain.trainer._trainer import Trainer as Trainer


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
