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
from jaxtyping import Array, PyTree

from rltrain.agents.vanilla_dqn import VanillaDQN
from rltrain.transitions import Transition


class DoubleDQN(VanillaDQN):
    r"""Double DQN — online network selects, target network evaluates.

    Inherits all behaviour from :class:`VanillaDQN`. Overriding ``_td_errors``
    (rather than ``_loss`` directly) means the inherited ``_loss`` and
    ``_loss_weighted`` automatically pick up the double-Q target — uniform
    Bellman MSE and PER-aware IS-weighted MSE both compose cleanly.
    """

    def _td_errors(
        self,
        target_params: PyTree[Array],
        static: PyTree,
        batch: Transition,
    ) -> Array:
        r"""Per-sample TD errors with double-Q target.

        $$\delta_i = r_i + \gamma\,Q_{\text{target}}\!\bigl(s'_i,\;
        \arg\max_{a'} Q_{\text{online}}(s'_i, a')\bigr)\,(1 - d_i) - Q(s_i, a_i)$$
        """
        target_net = eqx.combine(target_params, static).q_net

        q_all = jax.vmap(self.q_net)(batch.obs)
        q_sa = q_all[jnp.arange(q_all.shape[0]), batch.action.astype(jnp.int32)]

        # Online network SELECTS the best next action; target EVALUATES it.
        online_q_next = jax.vmap(self.q_net)(batch.next_obs)
        best_actions = jnp.argmax(online_q_next, axis=-1)
        target_q_next = jax.vmap(target_net)(batch.next_obs)
        target_max = target_q_next[jnp.arange(target_q_next.shape[0]), best_actions]

        td_target = batch.reward + self.gamma * target_max * (1.0 - batch.done.astype(jnp.float32))
        return td_target - q_sa
