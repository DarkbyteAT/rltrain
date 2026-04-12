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

from spike.agents.agent import TrainState, gradient_step, init_target_params
from spike.networks import MLP
from spike.transitions import Transition


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
        q_net: Q-network architecture (MLP).
        optimizer: Optax gradient transformation.
        gamma: Discount factor $\gamma \in [0, 1]$.
        target_rate: Polyak averaging coefficient $\tau$ for target updates.
        num_actions: Number of discrete actions.
        eps_start: Initial exploration epsilon.
        eps_end: Final exploration epsilon.
        eps_decay: Epsilon decay per learn step.
    """

    q_net: MLP
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

    def learn(self, state: DQNState, batch: Transition) -> tuple[DQNState, dict[str, Float[Array, ""]]]:
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
            return agent._loss(state.target_params, static, batch)

        new_params, new_opt_state, loss_val = gradient_step(loss_fn, state.params, state.opt_state, self.optimizer)

        # Polyak averaging: target ← τ·online + (1-τ)·target
        new_target_params = optax.incremental_update(new_params, state.target_params, self.target_rate)

        # Epsilon decay
        new_epsilon = jnp.maximum(
            jnp.array(self.eps_end),
            state.epsilon - jnp.array(self.eps_decay),
        )

        new_state = DQNState(
            params=new_params,
            opt_state=new_opt_state,
            target_params=new_target_params,
            epsilon=new_epsilon,
        )
        return new_state, {"loss": loss_val}

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

    # --------------- Internal ---------------

    def _loss(
        self,
        target_params: PyTree[Array],
        static: PyTree,
        batch: Transition,
    ) -> Float[Array, ""]:
        r"""Bellman MSE loss.

        $$L = \frac{1}{B}\sum_i \bigl(r_i + \gamma \max_{a'} Q_{\text{target}}(s'_i, a')
        \cdot (1 - d_i) - Q(s_i, a_i)\bigr)^2$$

        Called on a reconstructed agent (with online params combined), so
        ``self.q_net`` carries the online weights.
        """
        target_net = eqx.combine(target_params, static).q_net

        q_all = jax.vmap(self.q_net)(batch.obs)
        q_sa = q_all[jnp.arange(q_all.shape[0]), batch.action.astype(jnp.int32)]

        target_q_all = jax.vmap(target_net)(batch.next_obs)
        target_max = jnp.max(target_q_all, axis=-1)

        td_target = batch.reward + self.gamma * target_max * (1.0 - batch.done.astype(jnp.float32))
        td_error = td_target - q_sa
        return jnp.mean(td_error**2)
