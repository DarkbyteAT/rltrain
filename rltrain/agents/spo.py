r"""Surrogate Policy Optimisation (SPO) — extends PPO with a quadratic penalty.

Replaces PPO's clipped surrogate with a smooth quadratic penalty:

$$\mathcal{L}^{\mathrm{SPO}}(\theta) = -\mathbb{E}\!\bigl[
    r_t(\theta)\,A_t - \frac{|A_t|}{2\varepsilon}\,(r_t(\theta) - 1)^2
\bigr]$$

The quadratic term penalises large ratio deviations proportionally to the
magnitude of the advantage, providing a differentiable alternative to
PPO's hard clipping.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from rltrain.agents.agent import OnPolicyAgent, TrainState, gradient_step
from rltrain.math import gae
from rltrain.networks import MLP
from rltrain.transitions import Transition


class SPO(OnPolicyAgent):
    r"""SPO agent with quadratic penalty surrogate and mini-batch epochs.

    Inherits ``init`` and ``act`` from :class:`OnPolicyAgent`.
    Overrides ``learn`` for the multi-epoch mini-batch loop.

    Identical to PPO except for the surrogate objective: instead of clipping
    the ratio, SPO applies a smooth quadratic penalty
    $|A_t|/(2\varepsilon) \cdot (r_t - 1)^2$ that grows with the advantage
    magnitude.
    """

    critic: MLP
    gamma: float = eqx.field(static=True)
    tau: float = eqx.field(static=True)
    beta_critic: float = eqx.field(static=True)
    lambda_gae: float = eqx.field(static=True)
    eps_clip: float = eqx.field(static=True)
    num_epochs: int = eqx.field(static=True)
    minibatch_size: int = eqx.field(static=True)

    # --------------- Protocol methods ---------------

    def learn(
        self, state: TrainState, batch: Transition, key: PRNGKeyArray
    ) -> tuple[TrainState, dict[str, Float[Array, ""]]]:
        r"""SPO learning step: GAE, then multiple epochs of mini-batch quadratic-penalty updates."""
        static = eqx.partition(self, eqx.is_array)[1]

        # 1. Compute old log-probs (frozen)
        agent_old = eqx.combine(state.params, static)
        old_features = jax.vmap(agent_old.actor)(batch.obs)
        old_dists = jax.vmap(agent_old.action_head)(old_features)
        old_log_probs = jax.lax.stop_gradient(old_dists.log_prob(batch.action))

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
            key, epoch_key = jax.random.split(key)
            perm = jax.random.permutation(epoch_key, horizon_size)

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
                    return agent._spo_loss(_mb, _mb_old_lp, _mb_adv, _mb_ret)

                params, opt_state, loss_val = gradient_step(loss_fn, params, opt_state, self.optimizer)
                total_loss = total_loss + loss_val

        new_state = TrainState(
            params=params,
            opt_state=opt_state,
            target_params=state.target_params,
        )
        n_steps = jnp.array(self.num_epochs * num_minibatches, dtype=jnp.float32)
        return new_state, {"loss": total_loss / jnp.maximum(n_steps, 1.0)}

    # --------------- Internal ---------------

    def _spo_loss(
        self,
        transitions: Transition,
        old_log_probs: Float[Array, " T"],
        advantages: Float[Array, " T"],
        returns: Float[Array, " T"],
    ) -> Float[Array, ""]:
        r"""Quadratic penalty surrogate loss for a mini-batch.

        $$\mathcal{L} = -\frac{1}{T}\sum_t \bigl[r_t A_t
          - \frac{|A_t|}{2\varepsilon}(r_t - 1)^2\bigr]
          + \beta_c (R_t - V(s_t))^2 - \tau H[\pi]$$
        """
        # Current policy
        features = jax.vmap(self.actor)(transitions.obs)
        dists = jax.vmap(self.action_head)(features)
        log_probs = dists.log_prob(transitions.action)
        entropy = dists.entropy()

        # Probability ratio
        ratio = jnp.exp(log_probs - old_log_probs)

        # SPO surrogate: r*A - |A|/(2*eps) * (r-1)^2
        penalty = jnp.abs(advantages) / (2.0 * self.eps_clip) * (ratio - 1.0) ** 2
        actor_loss = -jnp.mean(ratio * advantages - penalty)

        # Critic loss
        values = jax.vmap(lambda o: self.critic(o).squeeze(-1))(transitions.obs)
        critic_loss = jnp.mean((returns - values) ** 2)

        # Entropy bonus
        entropy_loss = -self.tau * jnp.mean(entropy)

        return actor_loss + self.beta_critic * critic_loss + entropy_loss
