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
import optax
from jaxtyping import Array, Float, PRNGKeyArray

from spike.agents.agent import TrainState, gradient_step, zero_target_params
from spike.heads import DiscreteHead
from spike.math import gae
from spike.networks import MLP
from spike.transitions import Transition


class AdvantageAC(eqx.Module):
    r"""Advantage Actor-Critic with GAE and horizon-based collection.

    Uses Generalised Advantage Estimation for lower-variance advantage
    signals.  The $\lambda$ parameter controls the bias-variance trade-off
    between TD(0) ($\lambda=0$) and Monte Carlo ($\lambda=1$).
    """

    actor: MLP
    action_head: DiscreteHead
    critic: MLP
    optimizer: optax.GradientTransformation = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    tau: float = eqx.field(static=True)
    beta_critic: float = eqx.field(static=True)
    lambda_gae: float = eqx.field(static=True)

    # --------------- Protocol methods ---------------

    def init(self, key: PRNGKeyArray) -> TrainState:
        """Construct the initial training state."""
        params, _static = eqx.partition(self, eqx.is_array)
        opt_state = self.optimizer.init(params)
        target_params = zero_target_params(params)
        return TrainState(params=params, opt_state=opt_state, target_params=target_params)

    def learn(self, state: TrainState, batch: Transition) -> tuple[TrainState, dict[str, Float[Array, ""]]]:
        r"""One gradient step on the A2C loss with GAE advantages."""
        static = eqx.partition(self, eqx.is_array)[1]

        def loss_fn(params):
            agent = eqx.combine(params, static)
            return agent._loss(batch)

        new_params, new_opt_state, loss_val = gradient_step(loss_fn, state.params, state.opt_state, self.optimizer)
        new_state = TrainState(
            params=new_params,
            opt_state=new_opt_state,
            target_params=state.target_params,
        )
        return new_state, {"loss": loss_val}

    def act(self, state: TrainState, obs: Float[Array, " d"], key: PRNGKeyArray) -> Array:
        """Sample an action from the policy."""
        static = eqx.partition(self, eqx.is_array)[1]
        agent = eqx.combine(state.params, static)
        features = agent.actor(obs)
        dist = agent.action_head(features)
        return dist.sample(key)

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
        log_probs = dists.log_prob(transitions.action.squeeze(-1))
        entropy = dists.entropy()

        actor_loss = -jnp.mean(log_probs * advantages)
        critic_loss = jnp.mean((returns - values) ** 2)
        entropy_loss = -self.tau * jnp.mean(entropy)

        return actor_loss + self.beta_critic * critic_loss + entropy_loss
