r"""Vanilla policy gradient (REINFORCE) as a pure-functional JAX agent.

The REINFORCE loss is

$$\mathcal{L}(\theta) = -\mathbb{E}\!\bigl[\log \pi_\theta(a|s)\,G_t\bigr]
                        - \tau\,\mathbb{E}\!\bigl[H[\pi_\theta(\cdot|s)]\bigr]$$

where $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$ are the discounted returns
and $\tau$ controls the entropy bonus.

The agent module is **static** — it holds network architecture, hyperparameters,
and the optimizer.  All trainable state lives in a ``TrainState`` pytree.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from rltrain.agents.agent import OnPolicyAgent
from rltrain.math import discount
from rltrain.transitions import Transition


class VanillaPG(OnPolicyAgent):
    r"""Vanilla policy gradient agent as a static Equinox module.

    Inherits ``init``, ``learn``, and ``act`` from :class:`OnPolicyAgent`.
    Only ``_loss`` is defined here.
    """

    gamma: float
    tau: float
    normalise: bool

    def _loss(self, transitions: Transition) -> Float[Array, ""]:
        r"""REINFORCE loss over a batch of transitions.

        $$-\frac{1}{T}\sum_t \log\pi(a_t|s_t)\,G_t
          \;-\;\tau\,\frac{1}{T}\sum_t H[\pi(\cdot|s_t)]$$
        """
        returns = discount(transitions.reward, transitions.done.astype(jnp.float32), self.gamma)

        if self.normalise:
            returns = (returns - jnp.mean(returns)) / (jnp.std(returns) + 1e-8)

        features = jax.vmap(self.actor)(transitions.obs)
        dists = jax.vmap(self.action_head)(features)

        log_probs = dists.log_prob(transitions.action)
        entropy = dists.entropy()

        actor_loss = -jnp.mean(log_probs * returns)
        entropy_loss = -self.tau * jnp.mean(entropy)
        return actor_loss + entropy_loss
