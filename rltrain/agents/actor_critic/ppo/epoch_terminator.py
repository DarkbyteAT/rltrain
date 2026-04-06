"""EpochTerminator protocol for controlling PPO's multi-epoch update loop.

Provides a composable extension point for deciding when to stop iterating
over epochs early.  The protocol is scoped to agents with multi-epoch
mini-batch training (currently PPO) — single-pass agents have no epochs
to terminate.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EpochTerminator(Protocol):
    """Predicate that decides whether to stop PPO's epoch loop early.

    Implementations must be pure predicates — they signal "stop" but do not
    reach into agent internals.  Whether to rollback parameters is a property
    of the implementation, read by PPO's loop logic.
    """

    rollback: bool

    def should_stop(self, approx_kl: float) -> bool:
        """Return ``True`` if the epoch loop should terminate.

        Parameters
        ----------
        `approx_kl` : `float`
            Approximate KL divergence between the collection policy and the
            current policy, computed from the last mini-batch's log ratios.
        """
        ...
