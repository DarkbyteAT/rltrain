r"""RL math primitives — pure functions for discounting, advantage estimation, and distributional RL.

All functions are jittable and composable with ``jax.grad``, ``vmap``, and ``lax.scan``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


# ---------------------------------------------------------------------------
# Whitening and interpolation
# ---------------------------------------------------------------------------


def center(x: Float[Array, ...], eps: float = 1e-8) -> Float[Array, ...]:
    r"""Standardise to zero mean and unit variance: $(x - \mu) / (\sigma + \epsilon)$."""
    return (x - jnp.mean(x)) / (jnp.std(x) + eps)


def lerp(x: Float[Array, ...], y: Float[Array, ...], tau: float) -> Float[Array, ...]:
    r"""Polyak interpolation $\tau x + (1 - \tau) y$ — used for target network soft updates."""
    return tau * x + (1.0 - tau) * y


# ---------------------------------------------------------------------------
# Discounted returns
# ---------------------------------------------------------------------------


def discount(
    rewards: Float[Array, " T"],
    dones: Float[Array, " T"],
    gamma: float,
) -> Float[Array, " T"]:
    r"""Compute discounted returns $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$.

    Uses ``jax.lax.scan`` in reverse so the operation is jittable.
    Episode boundaries (``dones == 1``) reset the accumulator.
    """

    def _step(
        acc: Float[Array, ""],
        xs: tuple[Float[Array, ""], Float[Array, ""]],
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        r, d = xs
        acc = r + gamma * acc * (1.0 - d)
        return acc, acc

    _, returns = jax.lax.scan(_step, jnp.zeros(()), (rewards, dones), reverse=True)
    return returns


# ---------------------------------------------------------------------------
# Generalised Advantage Estimation (GAE)
# ---------------------------------------------------------------------------


def gae(
    values: Float[Array, " T_plus_1"],
    rewards: Float[Array, " T"],
    dones: Float[Array, " T"],
    gamma: float,
    lambda_gae: float,
) -> tuple[Float[Array, " T"], Float[Array, " T"]]:
    r"""Compute GAE advantages and returns via reverse ``lax.scan``.

    The TD residuals are

    $$\delta_t = r_t + \gamma\,V(s_{t+1})\,(1 - d_t) - V(s_t)$$

    and the advantages are the exponentially-weighted sum

    $$A_t = \sum_{k=0}^{T-t} (\gamma\lambda)^k \delta_{t+k}$$

    Args:
        values: Value estimates of length ``T+1``.  ``values[T]`` is the
            bootstrap value $V(s_{T+1})$ — pass zero if the episode ended.
        rewards: Rewards of length ``T``.
        dones: Done flags of length ``T`` (1.0 at episode boundaries).
        gamma: Discount factor $\gamma \in [0, 1]$.
        lambda_gae: GAE mixing parameter $\lambda \in [0, 1]$.

    Returns:
        ``(advantages, returns)`` both of length ``T``, where
        ``returns = advantages + values[:-1]``.

    Note:
        Advantages used in the actor loss **must** be wrapped in
        ``jax.lax.stop_gradient`` to prevent the actor gradient from
        flowing through the critic's value estimates.
    """
    values_t = values[:-1]  # V(s_t), shape (T,)
    values_tp1 = values[1:]  # V(s_{t+1}), shape (T,)

    deltas = rewards + gamma * values_tp1 * (1.0 - dones) - values_t

    def _step(
        acc: Float[Array, ""],
        xs: tuple[Float[Array, ""], Float[Array, ""]],
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        delta, done = xs
        acc = delta + gamma * lambda_gae * acc * (1.0 - done)
        return acc, acc

    _, advantages = jax.lax.scan(_step, jnp.zeros(()), (deltas, dones), reverse=True)
    returns = advantages + values_t
    return advantages, returns


# ---------------------------------------------------------------------------
# Distributional RL helpers (C51)
# ---------------------------------------------------------------------------


def q_values_from_pmf(
    pmf: Float[Array, "... num_atoms"],
    atoms: Float[Array, " num_atoms"],
) -> Float[Array, ...]:
    r"""Compute expected Q-values from a categorical distribution over atoms.

    $$Q(s, a) = \sum_i z_i \, p_i(s, a)$$

    Args:
        pmf: Probability mass function over atoms.  Trailing axis is the
            atom dimension; leading axes are batch and/or action dimensions.
        atoms: Fixed support vector $[V_{\min}, \ldots, V_{\max}]$.

    Returns:
        Expected Q-values with the atom dimension contracted.
    """
    return jnp.sum(pmf * atoms, axis=-1)


def project_distribution(
    target_pmf: Float[Array, "B num_atoms"],
    rewards: Float[Array, " B"],
    dones: Float[Array, " B"],
    gamma: float,
    atoms: Float[Array, " num_atoms"],
) -> Float[Array, "B num_atoms"]:
    r"""Distributional Bellman projection onto fixed atom support.

    Shifts each target atom by $r + \gamma z_j$ (zeroing the discount at
    episode boundaries), clips to $[V_{\min}, V_{\max}]$, and distributes
    probability mass to the two nearest atoms via linear interpolation.

    Args:
        target_pmf: Target network's PMF for the selected action, shape ``(B, N)``.
        rewards: Batch rewards, shape ``(B,)``.
        dones: Done flags, shape ``(B,)``.
        gamma: Discount factor.
        atoms: Fixed support vector of length ``N``.

    Returns:
        Projected PMF of shape ``(B, N)`` summing to 1 along the atom axis.
    """
    num_atoms = atoms.shape[0]
    v_min, v_max = atoms[0], atoms[-1]
    delta_z = (v_max - v_min) / (num_atoms - 1)

    # Bellman shift: T_z = r + γ·z (clipped to support)
    tz = rewards[:, None] + gamma * (1.0 - dones[:, None]) * atoms[None, :]
    tz = jnp.clip(tz, v_min, v_max)

    # Fractional atom indices
    b_idx = (tz - v_min) / delta_z  # (B, N), continuous indices
    lower = jnp.floor(b_idx).astype(jnp.int32)
    upper = jnp.ceil(b_idx).astype(jnp.int32)

    # Clamp to valid range (handles edge case where b_idx is exactly an integer)
    lower = jnp.clip(lower, 0, num_atoms - 1)
    upper = jnp.clip(upper, 0, num_atoms - 1)

    # Linear interpolation weights
    upper_weight = b_idx - lower.astype(jnp.float32)  # fraction toward upper
    lower_weight = 1.0 - upper_weight

    # Scatter probability mass to lower and upper atoms
    batch_size = target_pmf.shape[0]
    projected = jnp.zeros((batch_size, num_atoms))

    batch_idx = jnp.arange(batch_size)[:, None]  # (B, 1)

    # Add mass to lower atoms
    projected = projected.at[batch_idx, lower].add(target_pmf * lower_weight)
    # Add mass to upper atoms
    projected = projected.at[batch_idx, upper].add(target_pmf * upper_weight)

    return projected
