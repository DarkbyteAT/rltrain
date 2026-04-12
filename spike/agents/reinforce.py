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
import optax
from jaxtyping import Array, Float, PRNGKeyArray

from spike.agents.agent import TrainState, gradient_step, zero_target_params
from spike.heads import DiscreteHead
from spike.math import discount
from spike.networks import MLP
from spike.transitions import Transition


class REINFORCE(eqx.Module):
    r"""REINFORCE with learned value baseline as a static Equinox module.

    Extends VanillaPG by adding a critic network that estimates $V(s)$.
    The advantage $G_t - V(s_t)$ replaces raw returns in the policy gradient,
    reducing variance.  A single optimizer updates both actor and critic
    via the combined loss.
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
        r"""One gradient step on the REINFORCE-with-baseline loss."""
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
        log_probs = dists.log_prob(transitions.action.squeeze(-1))
        entropy = dists.entropy()

        actor_loss = -jnp.mean(log_probs * advantages)
        critic_loss = jnp.mean((returns - values) ** 2)
        entropy_loss = -self.tau * jnp.mean(entropy)

        return actor_loss + self.beta_critic * critic_loss + entropy_loss
