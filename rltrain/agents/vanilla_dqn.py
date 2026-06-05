r"""Vanilla DQN agent — pure-functional JAX implementation.

Implements the Bellman MSE loss with epsilon-greedy exploration,
target network Polyak averaging, and fully external training state.

The agent module holds architecture and static hyperparameters only.
Mutable training state (params, target params, optimizer state, epsilon)
lives in a ``DQNState`` pytree, threaded through ``learn`` and ``act``.
"""

from __future__ import annotations

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from rltrain.agents.agent import (
    TrainState,
    default_act_batch,
    dqn_learn_step_with_aux,
    init_target_params,
)
from rltrain.transitions import Transition


# ---------------------------------------------------------------------------
# DQN-specific training state
# ---------------------------------------------------------------------------


@chex.dataclass
class DQNState(TrainState):
    """Training state for DQN agents, extending TrainState with epsilon."""

    epsilon: Float[Array, ""]


# ---------------------------------------------------------------------------
# Agent module (static — no trainable array leaves)
# ---------------------------------------------------------------------------


class VanillaDQN(eqx.Module):
    r"""Q-learning agent with a target network.

    The module stores network architecture and static hyperparameters.
    All mutable state — online params, target params, optimizer state,
    and epsilon — lives in a ``DQNState`` pytree.

    Args:
        q_net: Q-network architecture. Any ``eqx.Module`` that maps an
            observation to a vector of action values.
        optimizer: Optax gradient transformation.
        gamma: Discount factor $\gamma \in [0, 1]$.
        target_rate: Polyak averaging coefficient $\tau$ for target updates.
        num_actions: Number of discrete actions.
        eps_start: Initial exploration epsilon.
        eps_end: Final exploration epsilon.
        eps_decay: Epsilon decay per learn step.
    """

    q_net: eqx.Module
    optimizer: optax.GradientTransformation = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    target_rate: float = eqx.field(static=True)
    num_actions: int = eqx.field(static=True)
    eps_start: float = eqx.field(static=True)
    eps_end: float = eqx.field(static=True)
    eps_decay: float = eqx.field(static=True)

    # --------------- Protocol methods ---------------

    def init(self, key: PRNGKeyArray) -> DQNState:
        """Construct the initial training state.

        Target params are initialised as a copy of the online params.
        Epsilon starts at ``eps_start``.
        """
        params, _static = eqx.partition(self, eqx.is_array)
        opt_state = self.optimizer.init(params)
        target_params = init_target_params(params)
        return DQNState(
            params=params,
            opt_state=opt_state,
            target_params=target_params,
            epsilon=jnp.array(self.eps_start),
        )

    def learn(
        self, state: DQNState, batch: Transition, _key: PRNGKeyArray
    ) -> tuple[DQNState, dict[str, Float[Array, ""]]]:
        r"""One DQN learning step: gradient descent + Polyak target update.

        Computes the Bellman MSE loss, applies one optimizer step, soft-updates
        the target network, and decays epsilon.

        Args:
            state: Current training state.
            batch: Batched transitions sampled from the replay buffer.

        Returns:
            ``(new_state, metrics)`` with updated params, target, opt_state,
            and decayed epsilon.
        """
        static = eqx.partition(self, eqx.is_array)[1]

        def loss_fn(params):
            agent = eqx.combine(params, static)
            # _loss_weighted always returns (loss, {"td_errors": ...}).
            # is_weights is a sentinel of 1.0 on the uniform sampling path
            # (see Transition + buffer_sample), so the mathematics reduce
            # to uniform Bellman MSE; PER paths see the real IS weights.
            return agent._loss_weighted(state.target_params, static, batch, batch.is_weights)

        return dqn_learn_step_with_aux(
            loss_fn,
            state,
            self.optimizer,
            self.target_rate,
            self.eps_end,
            self.eps_decay,
        )

    def act(self, state: DQNState, obs: Float[Array, " obs_dim"], key: PRNGKeyArray) -> Array:
        r"""Epsilon-greedy action selection.

        With probability ``state.epsilon`` sample uniformly; otherwise
        $\arg\max_a Q(s,a)$.
        """
        static = eqx.partition(self, eqx.is_array)[1]
        agent = eqx.combine(state.params, static)
        q = agent.q_net(obs)
        best_action = jnp.argmax(q)
        key_choice, key_rand = jax.random.split(key)
        random_action = jax.random.randint(key_rand, (), 0, self.num_actions)
        return jnp.where(
            jax.random.uniform(key_choice) < state.epsilon,
            random_action,
            best_action,
        )

    def act_batch(self, state: DQNState, obs: Float[Array, "N obs_dim"], key: PRNGKeyArray) -> Array:
        """Epsilon-greedy action selection for a batch of N observations (default vmap)."""
        return default_act_batch(self, state, obs, key)

    # --------------- Internal ---------------

    def _td_errors(
        self,
        target_params: PyTree[Array],
        static: PyTree,
        batch: Transition,
    ) -> Float[Array, " B"]:
        r"""Compute per-sample TD errors.

        $$\delta_i = r_i + \gamma \max_{a'} Q_{\text{target}}(s'_i, a')
        \cdot (1 - d_i) - Q(s_i, a_i)$$

        Returns:
            TD errors with shape ``(batch_size,)``.
        """
        target_net = eqx.combine(target_params, static).q_net

        q_all = jax.vmap(self.q_net)(batch.obs)
        q_sa = q_all[jnp.arange(q_all.shape[0]), batch.action.astype(jnp.int32)]

        target_q_all = jax.vmap(target_net)(batch.next_obs)
        target_max = jnp.max(target_q_all, axis=-1)

        td_target = batch.reward + self.gamma * target_max * (1.0 - batch.done.astype(jnp.float32))
        return td_target - q_sa

    def _loss(
        self,
        target_params: PyTree[Array],
        static: PyTree,
        batch: Transition,
    ) -> Float[Array, ""]:
        r"""Bellman MSE loss (uniform weighting).

        $$L = \frac{1}{B}\sum_i \delta_i^2$$
        """
        td_error = self._td_errors(target_params, static, batch)
        return jnp.mean(td_error**2)

    def _loss_weighted(
        self,
        target_params: PyTree[Array],
        static: PyTree,
        batch: Transition,
        is_weights: Array,
    ) -> tuple[Float[Array, ""], dict[str, Array]]:
        r"""IS-weighted Bellman MSE loss with auxiliary TD errors for PER.

        $$L = \frac{1}{B}\sum_i w_i \, \delta_i^2$$

        where $w_i$ are importance-sampling weights from prioritised replay.

        Returns:
            ``(loss, {"td_errors": |delta|})`` — the aux dict carries absolute
            TD errors for priority updates.
        """
        td_error = self._td_errors(target_params, static, batch)
        per_sample_loss = td_error**2
        weighted_loss = jnp.mean(is_weights * per_sample_loss)
        return weighted_loss, {"td_errors": jnp.abs(td_error)}
