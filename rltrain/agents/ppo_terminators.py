r"""Composable epoch terminators for PPO and other multi-epoch agents.

An :class:`EpochTerminator` decides — based on metrics computed at the end of
an epoch — whether the agent should stop running further epochs of mini-batch
optimisation. Terminators are JAX-traceable: ``should_stop`` returns a scalar
boolean ``Array``, so PPO can short-circuit remaining epochs via
``jax.lax.cond`` without breaking JIT.

Currently provided: :class:`KLEarlyStop`, which halts when the approximate
KL divergence between the current and old policy exceeds ``kl_threshold``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


@runtime_checkable
class EpochTerminator(Protocol):
    """Decides whether to stop further epoch updates inside an agent's learn step."""

    def should_stop(self, metrics: dict[str, Float[Array, ""]]) -> Bool[Array, ""]:
        """Return True if the agent should stop running additional epochs."""
        ...


class KLEarlyStop(eqx.Module):
    r"""Stop further epochs when approximate KL divergence exceeds a threshold.

    The PPO trainer is expected to provide ``approx_kl`` in the metrics dict
    (mean of ``log_pi_old - log_pi_new`` over the minibatch — Schulman 2020's
    "approximate KL" estimator).

    Args:
        kl_threshold: Maximum tolerated approximate KL. When exceeded, the
            agent halts its remaining epoch updates.
    """

    kl_threshold: float = eqx.field(static=True)

    def should_stop(self, metrics: dict[str, Float[Array, ""]]) -> Bool[Array, ""]:
        """Return True when ``metrics['approx_kl']`` exceeds ``kl_threshold``."""
        approx_kl = metrics.get("approx_kl", jnp.zeros(()))
        return approx_kl > self.kl_threshold
