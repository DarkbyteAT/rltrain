r"""Advantage Actor-Critic (A2C) — extends VanillaAC with GAE.

Replaces the one-step TD error with Generalised Advantage Estimation (GAE):

$$A_t = \sum_{k=0}^{T-t} (\gamma\lambda)^k \delta_{t+k}$$

where $\delta_t = r_t + \gamma V(s_{t+1})(1-d_t) - V(s_t)$.  The mixing
parameter $\lambda$ interpolates between bias (low $\lambda$) and variance
(high $\lambda$).  Horizon-based collection replaces episode-based.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from rltrain.agents.agent import OnPolicyAgent
from rltrain.math import gae
from rltrain.networks import MLP
from rltrain.transitions import Transition


class AdvantageAC(OnPolicyAgent):
    r"""Advantage Actor-Critic with GAE and horizon-based collection.

    Inherits ``init``, ``learn``, and ``act`` from :class:`OnPolicyAgent`.
    Only ``_loss`` (and helper ``_compute_gae``) are defined here.

    Uses Generalised Advantage Estimation for lower-variance advantage
    signals.  The $\lambda$ parameter controls the bias-variance trade-off
    between TD(0) ($\lambda=0$) and Monte Carlo ($\lambda=1$).
    """

    critic: MLP
    gamma: float = eqx.field(static=True)
    tau: float = eqx.field(static=True)
    beta_critic: float = eqx.field(static=True)
    lambda_gae: float = eqx.field(static=True)

    # --------------- Internal ---------------

    def _compute_gae(self, transitions: Transition) -> tuple[Float[Array, " T"], Float[Array, " T"]]:
        r"""Compute GAE advantages and returns from a horizon batch.

        Bootstraps $V(s_{T+1})$ from the last ``next_obs`` in the batch.
        Returns are stop-gradiented for use as critic targets.
        """
        values = jax.vmap(lambda o: self.critic(o).squeeze(-1))(transitions.obs)
        bootstrap_value = self.critic(transitions.next_obs[-1]).squeeze(-1)
        values_t_plus_1 = jnp.concatenate([values, bootstrap_value[None]])

        advantages, returns = gae(
            values_t_plus_1,
            transitions.reward,
            transitions.done.astype(jnp.float32),
            self.gamma,
            self.lambda_gae,
        )
        return advantages, returns

    def _loss(self, transitions: Transition) -> Float[Array, ""]:
        r"""Combined actor + critic loss with GAE advantages.

        $$-\frac{1}{T}\sum_t \log\pi(a_t|s_t)\,\mathrm{sg}(A_t^{\mathrm{GAE}})
          \;+\;\beta_c\,\frac{1}{T}\sum_t (R_t - V(s_t))^2
          \;-\;\tau\,\frac{1}{T}\sum_t H[\pi(\cdot|s_t)]$$
        """
        advantages, returns = self._compute_gae(transitions)
        advantages = jax.lax.stop_gradient(advantages)
        returns = jax.lax.stop_gradient(returns)

        values = jax.vmap(lambda o: self.critic(o).squeeze(-1))(transitions.obs)

        # Actor
        features = jax.vmap(self.actor)(transitions.obs)
        dists = jax.vmap(self.action_head)(features)
        log_probs = dists.log_prob(transitions.action)
        entropy = dists.entropy()

        actor_loss = -jnp.mean(log_probs * advantages)
        critic_loss = jnp.mean((returns - values) ** 2)
        entropy_loss = -self.tau * jnp.mean(entropy)

        return actor_loss + self.beta_critic * critic_loss + entropy_loss
