r"""REINFORCE with learned value baseline — extends VanillaPG with a critic.

The loss is

$$\mathcal{L}(\theta) = -\mathbb{E}\!\bigl[\log \pi_\theta(a|s)\,(G_t - V_\phi(s_t))\bigr]
                        + \beta_c\,\mathbb{E}\!\bigl[(G_t - V_\phi(s_t))^2\bigr]
                        - \tau\,\mathbb{E}\!\bigl[H[\pi_\theta(\cdot|s)]\bigr]$$

The value baseline $V_\phi(s)$ reduces variance without introducing bias.
Advantages (returns minus baseline) are stop-gradiented in the actor loss
to prevent the actor gradient from flowing through the critic.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from rltrain.agents.agent import OnPolicyAgent
from rltrain.math import discount
from rltrain.transitions import Transition


class REINFORCE(OnPolicyAgent):
    r"""REINFORCE with learned value baseline.

    Inherits ``init``, ``learn``, and ``act`` from :class:`OnPolicyAgent`.
    Only ``_loss`` is defined here. The ``critic`` field accepts any
    ``eqx.Module`` mapping observations to a scalar value estimate.
    """

    critic: eqx.Module
    gamma: float
    tau: float
    beta_critic: float

    def _loss(self, transitions: Transition) -> Float[Array, ""]:
        r"""Combined actor + critic loss with value baseline.

        $$-\frac{1}{T}\sum_t \log\pi(a_t|s_t)\,\mathrm{sg}(G_t - V(s_t))
          \;+\;\beta_c\,\frac{1}{T}\sum_t (G_t - V(s_t))^2
          \;-\;\tau\,\frac{1}{T}\sum_t H[\pi(\cdot|s_t)]$$
        """
        returns = discount(transitions.reward, transitions.done.astype(jnp.float32), self.gamma)

        # Critic value estimates
        values = jax.vmap(lambda o: self.critic(o).squeeze(-1))(transitions.obs)

        # Advantage = returns - baseline (stop-gradiented for actor loss)
        advantages = jax.lax.stop_gradient(returns - values)

        # Actor
        features = jax.vmap(self.actor)(transitions.obs)
        dists = jax.vmap(self.action_head)(features)
        log_probs = dists.log_prob(transitions.action)
        entropy = dists.entropy()

        actor_loss = -jnp.mean(log_probs * advantages)
        critic_loss = jnp.mean((returns - values) ** 2)
        entropy_loss = -self.tau * jnp.mean(entropy)

        return actor_loss + self.beta_critic * critic_loss + entropy_loss
