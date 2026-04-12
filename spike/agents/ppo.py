r"""Proximal Policy Optimisation (PPO) — extends AdvantageAC with clipped surrogate.

The clipped surrogate objective is

$$\mathcal{L}^{\mathrm{CLIP}}(\theta) = -\mathbb{E}\!\bigl[
    \min\bigl(r_t(\theta)\,A_t,\;
    \mathrm{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\,A_t\bigr)
\bigr]$$

where $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{\mathrm{old}}}(a_t|s_t)$
is the probability ratio.  Mini-batch epochs over the horizon buffer allow
multiple gradient steps per data collection phase.
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


class PPO(eqx.Module):
    r"""PPO agent with clipped surrogate objective and mini-batch epochs.

    Extends AdvantageAC by:
    1. Computing old log-probs before the epoch loop (then stop-gradienting).
    2. Using the clipped surrogate ratio $r(\theta) = \exp(\log\pi - \log\pi_{\mathrm{old}})$.
    3. Running multiple epochs of mini-batch gradient steps over the horizon.
    """

    actor: MLP
    action_head: DiscreteHead
    critic: MLP
    optimizer: optax.GradientTransformation = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    tau: float = eqx.field(static=True)
    beta_critic: float = eqx.field(static=True)
    lambda_gae: float = eqx.field(static=True)
    eps_clip: float = eqx.field(static=True)
    num_epochs: int = eqx.field(static=True)
    minibatch_size: int = eqx.field(static=True)

    # --------------- Protocol methods ---------------

    def init(self, key: PRNGKeyArray) -> TrainState:
        """Construct the initial training state."""
        params, _static = eqx.partition(self, eqx.is_array)
        opt_state = self.optimizer.init(params)
        target_params = zero_target_params(params)
        return TrainState(params=params, opt_state=opt_state, target_params=target_params)

    def learn(self, state: TrainState, batch: Transition) -> tuple[TrainState, dict[str, Float[Array, ""]]]:
        r"""PPO learning step: GAE, then multiple epochs of mini-batch clipped updates.

        Args:
            state: Current training state.
            batch: Horizon batch of transitions.

        Returns:
            Updated state and metrics dict.
        """
        static = eqx.partition(self, eqx.is_array)[1]

        # 1. Compute old log-probs (frozen)
        agent_old = eqx.combine(state.params, static)
        old_features = jax.vmap(agent_old.actor)(batch.obs)
        old_dists = jax.vmap(agent_old.action_head)(old_features)
        old_log_probs = jax.lax.stop_gradient(old_dists.log_prob(batch.action.squeeze(-1)))

        # 2. Compute values and GAE
        values = jax.vmap(lambda o: agent_old.critic(o).squeeze(-1))(batch.obs)
        bootstrap_value = agent_old.critic(batch.next_obs[-1]).squeeze(-1)
        values_t_plus_1 = jnp.concatenate([values, bootstrap_value[None]])

        advantages, returns = gae(
            values_t_plus_1,
            batch.reward,
            batch.done.astype(jnp.float32),
            self.gamma,
            self.lambda_gae,
        )
        advantages = jax.lax.stop_gradient(advantages)
        returns = jax.lax.stop_gradient(returns)

        # Normalise advantages
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

        # 3. Epoch loop over mini-batches
        params = state.params
        opt_state = state.opt_state
        total_loss = jnp.zeros(())

        horizon_size = batch.obs.shape[0]
        num_minibatches = horizon_size // self.minibatch_size

        for _epoch in range(self.num_epochs):
            # Shuffle indices
            # Use a deterministic key derived from params to avoid needing a PRNG argument
            perm = jnp.arange(horizon_size)

            for mb_idx in range(num_minibatches):
                start = mb_idx * self.minibatch_size
                end = start + self.minibatch_size
                idx = perm[start:end]

                mb = jax.tree.map(lambda x, i=idx: x[i], batch)
                mb_old_lp = old_log_probs[idx]
                mb_adv = advantages[idx]
                mb_ret = returns[idx]

                def loss_fn(p, *, _mb=mb, _mb_old_lp=mb_old_lp, _mb_adv=mb_adv, _mb_ret=mb_ret):
                    agent = eqx.combine(p, static)
                    return agent._ppo_loss(_mb, _mb_old_lp, _mb_adv, _mb_ret)

                params, opt_state, loss_val = gradient_step(loss_fn, params, opt_state, self.optimizer)
                total_loss = total_loss + loss_val

        new_state = TrainState(
            params=params,
            opt_state=opt_state,
            target_params=state.target_params,
        )
        n_steps = jnp.array(self.num_epochs * num_minibatches, dtype=jnp.float32)
        return new_state, {"loss": total_loss / jnp.maximum(n_steps, 1.0)}

    def act(self, state: TrainState, obs: Float[Array, " d"], key: PRNGKeyArray) -> Array:
        """Sample an action from the policy."""
        static = eqx.partition(self, eqx.is_array)[1]
        agent = eqx.combine(state.params, static)
        features = agent.actor(obs)
        dist = agent.action_head(features)
        return dist.sample(key)

    # --------------- Internal ---------------

    def _ppo_loss(
        self,
        transitions: Transition,
        old_log_probs: Float[Array, " T"],
        advantages: Float[Array, " T"],
        returns: Float[Array, " T"],
    ) -> Float[Array, ""]:
        r"""Clipped surrogate loss for a mini-batch.

        $$\mathcal{L} = -\frac{1}{T}\sum_t \min(r_t A_t, \mathrm{clip}(r_t) A_t)
          + \beta_c (R_t - V(s_t))^2 - \tau H[\pi]$$
        """
        # Current policy
        features = jax.vmap(self.actor)(transitions.obs)
        dists = jax.vmap(self.action_head)(features)
        log_probs = dists.log_prob(transitions.action.squeeze(-1))
        entropy = dists.entropy()

        # Probability ratio
        ratio = jnp.exp(log_probs - old_log_probs)
        clipped_ratio = jnp.clip(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)

        # Clipped surrogate
        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        actor_loss = -jnp.mean(jnp.minimum(surr1, surr2))

        # Critic loss
        values = jax.vmap(lambda o: self.critic(o).squeeze(-1))(transitions.obs)
        critic_loss = jnp.mean((returns - values) ** 2)

        # Entropy bonus
        entropy_loss = -self.tau * jnp.mean(entropy)

        return actor_loss + self.beta_critic * critic_loss + entropy_loss
