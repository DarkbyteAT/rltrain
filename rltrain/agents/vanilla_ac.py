r"""Vanilla Actor-Critic — extends REINFORCE with TD error advantage.

Instead of Monte Carlo returns $G_t$, the advantage is estimated via the
one-step TD error:

$$A_t = r_t + \gamma\,V(s_{t+1})\,(1 - d_t) - V(s_t)$$

This introduces bias but has much lower variance than REINFORCE, enabling
learning from individual transitions rather than full episodes.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from rltrain.agents.agent import OnPolicyAgent
from rltrain.transitions import Transition


class VanillaAC(OnPolicyAgent):
    r"""Vanilla Actor-Critic with TD error advantage.

    Inherits ``init``, ``learn``, and ``act`` from :class:`OnPolicyAgent`.
    Only ``_loss`` is defined here.

    Replaces REINFORCE's Monte Carlo returns with the one-step TD error
    $\delta_t = r_t + \gamma V(s_{t+1})(1 - d_t) - V(s_t)$ as the advantage
    signal.  The critic loss regresses $V(s_t)$ toward the TD target
    $r_t + \gamma V(s_{t+1})$.
    """

    critic: eqx.Module
    gamma: float = eqx.field(static=True)
    tau: float = eqx.field(static=True)
    beta_critic: float = eqx.field(static=True)

    def _loss(self, transitions: Transition) -> Float[Array, ""]:
        r"""Combined actor + critic loss with TD error advantage.

        $$-\frac{1}{T}\sum_t \log\pi(a_t|s_t)\,\mathrm{sg}(\delta_t)
          \;+\;\beta_c\,\frac{1}{T}\sum_t \delta_t^2
          \;-\;\tau\,\frac{1}{T}\sum_t H[\pi(\cdot|s_t)]$$

        where $\delta_t = r_t + \gamma V(s_{t+1})(1-d_t) - V(s_t)$.
        """
        # Critic value estimates
        values = jax.vmap(lambda o: self.critic(o).squeeze(-1))(transitions.obs)
        next_values = jax.vmap(lambda o: self.critic(o).squeeze(-1))(transitions.next_obs)

        # TD error advantage (stop_gradient on target for semi-gradient TD)
        td_target = jax.lax.stop_gradient(
            transitions.reward + self.gamma * next_values * (1.0 - transitions.done.astype(jnp.float32))
        )
        td_error = td_target - values
        advantages = jax.lax.stop_gradient(td_error)

        # Actor
        features = jax.vmap(self.actor)(transitions.obs)
        dists = jax.vmap(self.action_head)(features)
        log_probs = dists.log_prob(transitions.action)
        entropy = dists.entropy()

        actor_loss = -jnp.mean(log_probs * advantages)
        critic_loss = jnp.mean(td_error**2)
        entropy_loss = -self.tau * jnp.mean(entropy)

        return actor_loss + self.beta_critic * critic_loss + entropy_loss
