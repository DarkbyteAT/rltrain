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
import optax
from jaxtyping import Array, Float, PRNGKeyArray

from spike.agents.agent import TrainState, gradient_step, zero_target_params
from spike.heads import DiscreteHead
from spike.networks import MLP
from spike.transitions import Transition


class VanillaAC(eqx.Module):
    r"""Vanilla Actor-Critic with TD error advantage.

    Replaces REINFORCE's Monte Carlo returns with the one-step TD error
    $\delta_t = r_t + \gamma V(s_{t+1})(1 - d_t) - V(s_t)$ as the advantage
    signal.  The critic loss regresses $V(s_t)$ toward the TD target
    $r_t + \gamma V(s_{t+1})$.
    """

    actor: MLP
    action_head: DiscreteHead
    critic: MLP
    optimizer: optax.GradientTransformation = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    tau: float = eqx.field(static=True)
    beta_critic: float = eqx.field(static=True)

    # --------------- Protocol methods ---------------

    def init(self, key: PRNGKeyArray) -> TrainState:
        """Construct the initial training state."""
        params, _static = eqx.partition(self, eqx.is_array)
        opt_state = self.optimizer.init(params)
        target_params = zero_target_params(params)
        return TrainState(params=params, opt_state=opt_state, target_params=target_params)

    def learn(self, state: TrainState, batch: Transition) -> tuple[TrainState, dict[str, Float[Array, ""]]]:
        r"""One gradient step on the actor-critic loss with TD error."""
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

        # TD error advantage
        td_target = transitions.reward + self.gamma * next_values * (1.0 - transitions.done.astype(jnp.float32))
        td_error = td_target - values
        advantages = jax.lax.stop_gradient(td_error)

        # Actor
        features = jax.vmap(self.actor)(transitions.obs)
        dists = jax.vmap(self.action_head)(features)
        log_probs = dists.log_prob(transitions.action.squeeze(-1))
        entropy = dists.entropy()

        actor_loss = -jnp.mean(log_probs * advantages)
        critic_loss = jnp.mean(td_error**2)
        entropy_loss = -self.tau * jnp.mean(entropy)

        return actor_loss + self.beta_critic * critic_loss + entropy_loss
