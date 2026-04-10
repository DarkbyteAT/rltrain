r"""Vanilla policy gradient (REINFORCE) as a pure-functional JAX agent.

The REINFORCE loss is

$$\mathcal{L}(\theta) = -\mathbb{E}\!\bigl[\log \pi_\theta(a|s)\,G_t\bigr]
                        - \tau\,\mathbb{E}\!\bigl[H[\pi_\theta(\cdot|s)]\bigr]$$

where $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$ are the discounted returns
and $\tau$ controls the entropy bonus.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray

from spike.heads import DiscreteHead
from spike.networks import MLP
from spike.transitions import Transition


# ---------------------------------------------------------------------------
# Discounted returns via reverse lax.scan
# ---------------------------------------------------------------------------


def discount(rewards: Float[Array, " T"], dones: Float[Array, " T"], gamma: float) -> Float[Array, " T"]:
    r"""Compute discounted returns $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$.

    Uses ``jax.lax.scan`` in reverse so the operation is jittable.
    Episode boundaries (``dones == 1``) reset the accumulator.
    """

    def _step(
        acc: Float[Array, ""], xs: tuple[Float[Array, ""], Float[Array, ""]]
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        r, d = xs
        acc = r + gamma * acc * (1.0 - d)
        return acc, acc

    _, returns = jax.lax.scan(
        _step,
        jnp.zeros(()),
        (rewards, dones),
        reverse=True,
    )
    return returns


# ---------------------------------------------------------------------------
# Agent module
# ---------------------------------------------------------------------------


class VanillaPG(eqx.Module):
    r"""Vanilla policy gradient agent as a pure Equinox module.

    Fields ``gamma``, ``tau``, and ``normalise`` are static (non-trainable)
    hyperparameters. The trainable parameters live inside ``actor`` and
    ``action_head``.
    """

    actor: MLP
    action_head: DiscreteHead
    gamma: float = eqx.field(static=True)
    tau: float = eqx.field(static=True)
    normalise: bool = eqx.field(static=True)

    def act(self, obs: Float[Array, " d"], key: PRNGKeyArray) -> tuple[Array, Float[Array, ""], Any]:
        """Forward pass: obs -> distribution -> (action, log_prob, dist).

        Pure function — no hidden state, no mutation.
        """
        features = self.actor(obs)
        dist = self.action_head(features)
        action = dist.sample(key)
        log_prob = dist.log_prob(action)
        return action, log_prob, dist

    def loss(self, transitions: Transition) -> Float[Array, ""]:
        r"""REINFORCE loss over a batch of transitions.

        $$-\frac{1}{T}\sum_t \log\pi(a_t|s_t)\,G_t
          \;-\;\tau\,\frac{1}{T}\sum_t H[\pi(\cdot|s_t)]$$
        """
        returns = discount(transitions.reward, transitions.done.astype(jnp.float32), self.gamma)

        if self.normalise:
            returns = (returns - jnp.mean(returns)) / (jnp.std(returns) + 1e-8)

        # Vectorised forward pass over the batch
        features = jax.vmap(self.actor)(transitions.obs)
        dists = jax.vmap(self.action_head)(features)

        log_probs = dists.log_prob(transitions.action.squeeze(-1))
        entropy = dists.entropy()

        actor_loss = -jnp.mean(log_probs * returns)
        entropy_loss = -self.tau * jnp.mean(entropy)
        return actor_loss + entropy_loss


# ---------------------------------------------------------------------------
# Module-level learn function (jittable)
# ---------------------------------------------------------------------------


def learn(
    params: Any,
    static: Any,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    transitions: Transition,
) -> tuple[Any, optax.OptState, dict[str, Float[Array, ""]]]:
    """One gradient step: compute loss, differentiate, apply optimizer.

    Args:
        params: Trainable parameters (from ``eqx.partition``).
        static: Static structure (from ``eqx.partition``).
        opt_state: Current optimizer state.
        optimizer: An optax optimizer.
        transitions: Batch of transitions for the update.

    Returns:
        ``(new_params, new_opt_state, metrics)`` where metrics contains
        the scalar loss.
    """
    agent = eqx.combine(params, static)

    loss_val, grads = eqx.filter_value_and_grad(lambda m: m.loss(transitions))(agent)

    # Extract only the parameter gradients (matching params structure)
    grad_params, _ = eqx.partition(grads, eqx.is_array)

    updates, new_opt_state = optimizer.update(grad_params, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, {"loss": loss_val}
