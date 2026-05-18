"""Carry types for the training loop.

Three dataclasses that separate loop-invariant config from mutable scan carry
and per-step output. ``TrainConfig`` is frozen Python; ``TrainCarry`` and
``StepOutput`` are chex dataclasses (pytrees) suitable for ``lax.scan``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chex
from jaxtyping import Array, PRNGKeyArray, PyTree

from rltrain.buffer import ExperienceBuffer


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Frozen loop-invariant training configuration.

    Bundles the ~11 scalar hyperparameters that every loop implementation
    needs, replacing a long kwargs signature on the TrainingLoop protocol.
    """

    num_steps: int
    checkpoint_steps: int
    collect_size: int
    min_buffer_size: int
    batch_size: int
    seed: int
    run_dir: Path | None


@chex.dataclass
class TrainCarry:
    """Monadic carry for ``lax.scan``.

    Used by ScanLoop and PmapLoop as the scan carry, and by PythonLoop
    to initialise local variables. All fields are pytrees of arrays.
    """

    agent_state: PyTree[Array]
    env_state: PyTree[Array]
    buffer: ExperienceBuffer
    step_count: Array  # int32 scalar
    key: PRNGKeyArray


@chex.dataclass
class StepOutput:
    """Per-step scan output for deferred callback dispatch.

    Accumulated over a checkpoint-sized segment by ``lax.scan``, then
    processed by Python code at segment boundaries to fire callbacks.
    """

    done: Array
    episode_return: Array
    episode_length: Array
    metrics: PyTree[Array]  # scalar metrics, keys fixed at trace time
    did_learn: Array
