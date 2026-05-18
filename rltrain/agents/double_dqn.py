r"""Double DQN agent — decouples action selection from evaluation.

Van Hasselt et al. (2016) showed that vanilla DQN overestimates Q-values
because the same network both selects and evaluates the greedy action.
Double DQN fixes this by using the **online** network to select the best
next action and the **target** network to evaluate it:

$$y_i = r_i + \gamma\,Q_{\text{target}}\!\bigl(s'_i,\;
        \arg\max_{a'} Q_{\text{online}}(s'_i, a')\bigr)\,(1 - d_i)$$

Everything else (learn, act, init, DQNState) is inherited unchanged from
:class:`VanillaDQN`.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from rltrain.agents.vanilla_dqn import VanillaDQN
from rltrain.transitions import Transition


class DoubleDQN(VanillaDQN):
    r"""Double DQN — online network selects, target network evaluates.

    Inherits all behaviour from :class:`VanillaDQN` and overrides only
    ``_loss`` to decouple action selection from evaluation.  The extra
    forward pass through the online network on ``next_obs`` is the sole
    difference.
    """

    def _loss(
        self,
        target_params: PyTree[Array],
        static: PyTree,
        batch: Transition,
    ) -> Float[Array, ""]:
        r"""Bellman MSE loss with double-Q target.

        $$L = \frac{1}{B}\sum_i \bigl(r_i + \gamma\,
        Q_{\text{target}}(s'_i, \arg\max_{a'} Q_{\text{online}}(s'_i, a'))
        \cdot (1 - d_i) - Q(s_i, a_i)\bigr)^2$$
        """
        target_net = eqx.combine(target_params, static).q_net

        # Online Q-values for current obs (for TD error)
        q_all = jax.vmap(self.q_net)(batch.obs)
        q_sa = q_all[jnp.arange(q_all.shape[0]), batch.action.astype(jnp.int32)]

        # Online network SELECTS the best next action
        online_q_next = jax.vmap(self.q_net)(batch.next_obs)
        best_actions = jnp.argmax(online_q_next, axis=-1)

        # Target network EVALUATES that action
        target_q_next = jax.vmap(target_net)(batch.next_obs)
        target_max = target_q_next[jnp.arange(target_q_next.shape[0]), best_actions]

        td_target = batch.reward + self.gamma * target_max * (1.0 - batch.done.astype(jnp.float32))
        td_error = td_target - q_sa
        return jnp.mean(td_error**2)
