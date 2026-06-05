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

from rltrain.agents.agent import default_act_batch, dqn_learn_step_with_aux, init_target_params
from rltrain.agents.vanilla_dqn import DQNState
from rltrain.math import project_distribution, q_values_from_pmf
from rltrain.transitions import Transition


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
        feature_net: Backbone network mapping observations to features. Any
            ``eqx.Module`` with a matching input/output shape.
        atom_head: Module mapping features to per-action PMFs. Conventionally
            a :class:`~rltrain.heads.CategoricalAtomHead`, but any
            ``eqx.Module`` exposing an ``atoms`` attribute and returning a
            ``(num_actions, num_atoms)`` PMF satisfies the contract.
        optimizer: Optax gradient transformation.
        gamma: Discount factor $\gamma \in [0, 1]$.
        target_rate: Polyak averaging coefficient $\tau$ for target updates.
        num_actions: Number of discrete actions.
        eps_start: Initial exploration epsilon.
        eps_end: Final exploration epsilon.
        eps_decay: Epsilon decay per learn step.
    """

    feature_net: eqx.Module
    atom_head: eqx.Module
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

    def learn(
        self, state: DQNState, batch: Transition, _key: PRNGKeyArray
    ) -> tuple[DQNState, dict[str, Float[Array, ""]]]:
        r"""One C51 learning step: IS-weighted cross-entropy + Polyak target update.

        Uniform sampling yields ``batch.is_weights = 1.0``, so the weighted
        path reduces to plain mean cross-entropy; prioritised sampling sees
        the real IS weights. ``td_errors`` (per-sample KL between projected
        target PMF and online PMF) is emitted in metrics for the trainer to
        route back into ``buffer.priorities``.
        """
        static = eqx.partition(self, eqx.is_array)[1]

        def loss_fn(params):
            agent = eqx.combine(params, static)
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

    def act_batch(self, state: DQNState, obs: Float[Array, "N obs_dim"], key: PRNGKeyArray) -> Array:
        """Epsilon-greedy action selection for a batch of N observations (default vmap)."""
        return default_act_batch(self, state, obs, key)

    # --------------- Internal ---------------

    def _forward(self, obs: Float[Array, " obs_dim"]) -> Float[Array, "num_actions num_atoms"]:
        """Forward pass: obs -> features -> per-action PMFs."""
        features = self.feature_net(obs)
        return self.atom_head(features)

    def _per_sample_xent(
        self,
        target_params: PyTree[Array],
        static: PyTree,
        batch: Transition,
    ) -> tuple[Float[Array, " B"], Float[Array, "B N"], Float[Array, "B N"]]:
        r"""Per-sample cross-entropy and projected target PMFs.

        Returns ``(per_sample_xent, online_pmf_sa, projected)`` — shared
        intermediates so ``_loss`` and ``_loss_weighted`` reuse the
        projection arithmetic.
        """
        target_agent = eqx.combine(target_params, static)
        atoms = self.atom_head.atoms

        online_pmf_all = jax.vmap(self._forward)(batch.obs)
        batch_idx = jnp.arange(online_pmf_all.shape[0])
        actions_int = batch.action.astype(jnp.int32)
        online_pmf_sa = online_pmf_all[batch_idx, actions_int]

        # Double-Q: online selects best next action; target evaluates.
        online_pmf_next = jax.vmap(self._forward)(batch.next_obs)
        online_q_next = q_values_from_pmf(online_pmf_next, atoms)
        best_next_actions = jnp.argmax(online_q_next, axis=-1)

        target_pmf_next = jax.vmap(target_agent._forward)(batch.next_obs)
        target_pmf_selected = target_pmf_next[batch_idx, best_next_actions]

        projected = project_distribution(
            target_pmf_selected,
            batch.reward,
            batch.done.astype(jnp.float32),
            self.gamma,
            atoms,
        )

        log_online = jnp.log(online_pmf_sa + 1e-8)
        per_sample_xent = -jnp.sum(projected * log_online, axis=-1)  # (B,)
        return per_sample_xent, online_pmf_sa, projected

    def _loss(
        self,
        target_params: PyTree[Array],
        static: PyTree,
        batch: Transition,
    ) -> Float[Array, ""]:
        r"""Cross-entropy loss with distributional Bellman projection (uniform).

        $$\mathcal{L} = -\frac{1}{B}\sum_b \sum_i \hat{m}_i \log p_i(s_b, a_b)$$
        """
        per_sample_xent, _, _ = self._per_sample_xent(target_params, static, batch)
        return jnp.mean(per_sample_xent)

    def _loss_weighted(
        self,
        target_params: PyTree[Array],
        static: PyTree,
        batch: Transition,
        is_weights: Array,
    ) -> tuple[Float[Array, ""], dict[str, Array]]:
        r"""IS-weighted cross-entropy loss with per-sample KL td_errors.

        Per-sample cross-entropy is weighted by ``is_weights`` then meaned:

        $$\mathcal{L} = \frac{1}{B}\sum_b w_b \cdot \mathrm{xent}_b$$

        ``td_errors`` are the per-sample KL divergence between the projected
        target PMF and the online predicted PMF — the standard Rainbow
        priority signal:

        $$\delta_b = \sum_i \hat{m}_i (\log \hat{m}_i - \log p_i(s_b, a_b))$$
        """
        per_sample_xent, online_pmf_sa, projected = self._per_sample_xent(target_params, static, batch)
        weighted_loss = jnp.mean(is_weights * per_sample_xent)

        # Per-sample KL(projected || online) for priority updates. Cross-entropy
        # already has -sum(projected * log(online)); subtracting the projected
        # self-entropy turns it into KL.
        projected_log = jnp.log(projected + 1e-8)
        projected_entropy = -jnp.sum(projected * projected_log, axis=-1)  # (B,)
        kl_per_sample = per_sample_xent - projected_entropy  # (B,) >= 0

        return weighted_loss, {"td_errors": kl_per_sample}
