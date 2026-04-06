"""KL divergence early stopping for PPO's epoch loop.

Implements KLE-Stop and KLE-Rollback from Dossa et al. (2021):

- **KLE-Stop** (``rollback=False``): stop epochs when KL exceeds
  the threshold; keep the parameters as-is.
- **KLE-Rollback** (``rollback=True``): stop epochs and restore
  parameters to their state before the offending epoch.

References
----------
Dossa et al., "An Empirical Investigation of Early Stopping
Optimizations in Proximal Policy Optimization", IEEE Access, 2021.
"""

from __future__ import annotations


class KLEarlyStop:
    """Stop PPO's epoch loop when approximate KL divergence exceeds a threshold.

    Parameters
    ----------
    `target_kl` : `float`
        KL divergence threshold.  When the epoch's approximate KL exceeds
        this value, ``should_stop`` returns ``True``.
    `rollback` : `bool`
        If ``True``, PPO restores parameters to their pre-epoch state
        when stopping (KLE-Rollback).  If ``False``, parameters stay
        as-is (KLE-Stop).
    """

    def __init__(self, target_kl: float, rollback: bool = True):
        self.target_kl = target_kl
        self.rollback = rollback

    def should_stop(self, approx_kl: float) -> bool:
        return approx_kl > self.target_kl
