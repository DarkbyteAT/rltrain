r"""Distributional DQN (C51) — learns the full value distribution.

Bellemare et al. (2017) replace the scalar Q-value with a categorical
distribution over a fixed set of atoms $\{z_i\}_{i=0}^{N-1}$ spanning
$[V_{\min}, V_{\max}]$.  The loss is the cross-entropy between the
projected Bellman distribution and the online distribution:

$$\mathcal{L} = -\sum_i \hat{m}_i \log p_i(s, a)$$

where $\hat{m}$ is the distributional Bellman projection of the target
PMF onto the atom support.

Action selection uses expected Q-values $Q(s,a) = \sum_i z_i\,p_i(s,a)$
with double-Q decoupling (online selects, target evaluates).

The network uses a :class:`CategoricalAtomHead` that outputs per-action
PMFs over the atom support, replacing VanillaDQN's scalar MLP output.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from spike.agents.agent import gradient_step, init_target_params
from spike.agents.vanilla_dqn import DQNState
from spike.heads import CategoricalAtomHead
from spike.math import project_distribution, q_values_from_pmf
from spike.networks import MLP
from spike.transitions import Transition


class DistributionalDQN(eqx.Module):
    r"""C51 distributional DQN agent.

    Instead of learning scalar Q-values, the network outputs a probability
    mass function over ``num_atoms`` fixed atoms per action.  The loss is
    cross-entropy between the projected target distribution and the online
    distribution.

    Action selection uses the expected Q-value (dot product of PMF with
    atoms) with double-Q decoupling: the online network selects the best
    action; the target network provides the distribution to project.

    Args:
        feature_net: Backbone MLP mapping observations to features.
        atom_head: CategoricalAtomHead mapping features to per-action PMFs.
        optimizer: Optax gradient transformation.
        gamma: Discount factor $\gamma \in [0, 1]$.
        target_rate: Polyak averaging coefficient $\tau$ for target updates.
        num_actions: Number of discrete actions.
        eps_start: Initial exploration epsilon.
        eps_end: Final exploration epsilon.
        eps_decay: Epsilon decay per learn step.
    """

    feature_net: MLP
    atom_head: CategoricalAtomHead
    optimizer: optax.GradientTransformation = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    target_rate: float = eqx.field(static=True)
    num_actions: int = eqx.field(static=True)
    eps_start: float = eqx.field(static=True)
    eps_end: float = eqx.field(static=True)
    eps_decay: float = eqx.field(static=True)

    # --------------- Protocol methods ---------------

    def init(self, key: PRNGKeyArray) -> DQNState:
        """Construct the initial training state."""
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
        r"""One C51 learning step: cross-entropy loss + Polyak target update."""
        static = eqx.partition(self, eqx.is_array)[1]

        def loss_fn(params):
            agent = eqx.combine(params, static)
            return agent._loss(state.target_params, static, batch)

        new_params, new_opt_state, loss_val = gradient_step(loss_fn, state.params, state.opt_state, self.optimizer)

        # Polyak averaging
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
        r"""Epsilon-greedy action selection using expected Q-values.

        Computes $Q(s,a) = \sum_i z_i\,p_i(s,a)$ and takes the argmax,
        with probability ``state.epsilon`` of a random action.
        """
        static = eqx.partition(self, eqx.is_array)[1]
        agent = eqx.combine(state.params, static)
        features = agent.feature_net(obs)
        pmf = agent.atom_head(features)  # (num_actions, num_atoms)
        q = q_values_from_pmf(pmf, agent.atom_head.atoms)  # (num_actions,)
        best_action = jnp.argmax(q)
        key_choice, key_rand = jax.random.split(key)
        random_action = jax.random.randint(key_rand, (), 0, self.num_actions)
        return jnp.where(
            jax.random.uniform(key_choice) < state.epsilon,
            random_action,
            best_action,
        )

    # --------------- Internal ---------------

    def _forward(self, obs: Float[Array, " obs_dim"]) -> Float[Array, "num_actions num_atoms"]:
        """Forward pass: obs -> features -> per-action PMFs."""
        features = self.feature_net(obs)
        return self.atom_head(features)

    def _loss(
        self,
        target_params: PyTree[Array],
        static: PyTree,
        batch: Transition,
    ) -> Float[Array, ""]:
        r"""Cross-entropy loss with distributional Bellman projection.

        1. Online PMFs for current obs -> select taken action's PMF
        2. Online PMFs for next obs -> compute Q-values -> select best action (double-Q)
        3. Target PMFs for next obs -> select best action's PMF
        4. Project target PMF via distributional Bellman operator
        5. Cross-entropy: $-\sum_i \hat{m}_i \log p_i$
        """
        target_agent = eqx.combine(target_params, static)
        atoms = self.atom_head.atoms

        # Online PMFs for current obs
        online_pmf_all = jax.vmap(self._forward)(batch.obs)  # (B, A, N)
        batch_idx = jnp.arange(online_pmf_all.shape[0])
        actions_int = batch.action.astype(jnp.int32)
        online_pmf_sa = online_pmf_all[batch_idx, actions_int]  # (B, N)

        # Double-Q: online network selects best next action
        online_pmf_next = jax.vmap(self._forward)(batch.next_obs)  # (B, A, N)
        online_q_next = q_values_from_pmf(online_pmf_next, atoms)  # (B, A)
        best_next_actions = jnp.argmax(online_q_next, axis=-1)  # (B,)

        # Target network evaluates: get PMF for selected action
        target_pmf_next = jax.vmap(target_agent._forward)(batch.next_obs)  # (B, A, N)
        target_pmf_selected = target_pmf_next[batch_idx, best_next_actions]  # (B, N)

        # Distributional Bellman projection
        projected = project_distribution(
            target_pmf_selected,
            batch.reward,
            batch.done.astype(jnp.float32),
            self.gamma,
            atoms,
        )  # (B, N)

        # Cross-entropy loss
        log_online = jnp.log(online_pmf_sa + 1e-8)
        loss = -jnp.sum(projected * log_online, axis=-1)  # (B,)
        return jnp.mean(loss)
