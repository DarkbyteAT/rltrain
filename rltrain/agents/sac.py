r"""Soft Actor-Critic (SAC) — pure-functional JAX implementation.

Supports both continuous (SquashedGaussianHead) and discrete (DiscreteHead)
action spaces via a ``discrete`` flag resolved at JIT trace time.

SAC uses three separate gradient steps per learn call:

1. **Critic** — twin Q-networks minimising Bellman MSE against a target
2. **Actor** — policy maximising expected Q minus entropy penalty
3. **Alpha** — automatic temperature tuning toward a target entropy

Each gradient step receives only the parameters it differentiates through.
Stop-gradient semantics are structural: parameters not in scope cannot
receive gradients.
"""

from __future__ import annotations

import math

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from rltrain.agents.agent import gradient_step, gradient_step_with_aux, init_target_params
from rltrain.heads import DiscreteHead
from rltrain.networks import MLP
from rltrain.transitions import Transition


LOG_EPS = 1e-8


# ---------------------------------------------------------------------------
# SAC-specific training state
# ---------------------------------------------------------------------------


@chex.dataclass
class SACState:
    r"""Training state for SAC, with separate param groups for the three losses.

    Fields:
        actor_params: Actor network + action head parameters.
        critic_params: Twin Q-network parameters (Q1 and Q2).
        log_alpha: Log temperature $\log\alpha$ (scalar array).
        actor_opt_state: Actor optimizer state.
        critic_opt_state: Critic optimizer state.
        alpha_opt_state: Alpha optimizer state.
        target_critic_params: Polyak-averaged target Q-network parameters.
    """

    actor_params: PyTree[Array]
    critic_params: PyTree[Array]
    log_alpha: Float[Array, ""]
    actor_opt_state: PyTree[Array]
    critic_opt_state: PyTree[Array]
    alpha_opt_state: PyTree[Array]
    target_critic_params: PyTree[Array]


# ---------------------------------------------------------------------------
# Agent module (static -- no trainable array leaves at trace time)
# ---------------------------------------------------------------------------


class SAC(eqx.Module):
    r"""Soft Actor-Critic with automatic entropy tuning.

    The module stores network architecture and static hyperparameters.
    All mutable state lives in ``SACState``.

    Args:
        actor: MLP backbone producing features for the action head.
        action_head: ``DiscreteHead`` or ``SquashedGaussianHead``.
        critic_1: First Q-network.
        critic_2: Second Q-network.
        actor_optimizer: Optax optimizer for the actor.
        critic_optimizer: Optax optimizer for the critics.
        alpha_optimizer: Optax optimizer for ``log_alpha``.
        gamma: Discount factor $\gamma \in [0, 1]$.
        tau: Polyak averaging rate for target critics.
        target_entropy: Target entropy for automatic alpha tuning.
        discrete: Whether the action space is discrete (resolved at trace time).
    """

    actor: MLP
    action_head: eqx.Module
    critic_1: MLP
    critic_2: MLP
    actor_optimizer: optax.GradientTransformation = eqx.field(static=True)
    critic_optimizer: optax.GradientTransformation = eqx.field(static=True)
    alpha_optimizer: optax.GradientTransformation = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    tau: float = eqx.field(static=True)
    target_entropy: float = eqx.field(static=True)
    discrete: bool = eqx.field(static=True)

    def __init__(
        self,
        actor: MLP,
        action_head: eqx.Module,
        critic_1: MLP,
        critic_2: MLP,
        actor_optimizer: optax.GradientTransformation,
        critic_optimizer: optax.GradientTransformation,
        alpha_optimizer: optax.GradientTransformation,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_entropy: float | None = None,
    ):
        """Initialise SAC, inferring ``discrete`` from the action head type."""
        self.actor = actor
        self.action_head = action_head
        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.alpha_optimizer = alpha_optimizer
        self.gamma = gamma
        self.tau = tau
        self.discrete = isinstance(action_head, DiscreteHead)

        if target_entropy is not None:
            self.target_entropy = float(target_entropy)
        elif self.discrete:
            num_actions = action_head.linear.out_features
            self.target_entropy = -math.log(1.0 / num_actions) * 0.98
        else:
            # Convention: -action_dim for continuous
            self.target_entropy = -float(action_head.gaussian.mu_linear.out_features)

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    def init(self, key: PRNGKeyArray) -> SACState:
        """Construct the initial training state with separate param groups."""
        actor_params, critic_params = self._partition_params()
        log_alpha = jnp.array(0.0)

        actor_opt_state = self.actor_optimizer.init(actor_params)
        critic_opt_state = self.critic_optimizer.init(critic_params)
        alpha_opt_state = self.alpha_optimizer.init(log_alpha)
        target_critic_params = init_target_params(critic_params)

        return SACState(
            actor_params=actor_params,
            critic_params=critic_params,
            log_alpha=log_alpha,
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
            alpha_opt_state=alpha_opt_state,
            target_critic_params=target_critic_params,
        )

    def learn(
        self,
        state: SACState,
        batch: Transition,
        key: PRNGKeyArray,
    ) -> tuple[SACState, dict[str, Float[Array, ""]]]:
        r"""One SAC learning step: critic, actor, and alpha updates.

        The three loss functions are differentiated with respect to separate
        parameter groups.  Stop-gradient is structural -- parameters not
        passed as arguments to a loss function cannot receive gradients.
        """
        k1, k2, k3 = jax.random.split(key, 3)

        _actor_static, _critic_static = self._statics()
        alpha = jnp.exp(state.log_alpha)

        # 1. Critic update
        def critic_loss_fn(critic_p):
            return self._critic_loss(
                critic_p,
                _critic_static,
                state.target_critic_params,
                state.actor_params,
                _actor_static,
                alpha,
                batch,
                k1,
            )

        new_critic, new_critic_opt, c_loss, critic_aux = gradient_step_with_aux(
            critic_loss_fn,
            state.critic_params,
            state.critic_opt_state,
            self.critic_optimizer,
        )

        # 2. Actor update (uses NEW critic params, closed over)
        def actor_loss_fn(actor_p):
            return self._actor_loss(
                actor_p,
                _actor_static,
                new_critic,
                _critic_static,
                alpha,
                batch,
                k2,
            )

        new_actor, new_actor_opt, a_loss = gradient_step(
            actor_loss_fn,
            state.actor_params,
            state.actor_opt_state,
            self.actor_optimizer,
        )

        # 3. Alpha update
        def alpha_loss_fn(log_a):
            return self._alpha_loss(
                log_a,
                new_actor,
                _actor_static,
                batch,
                k3,
            )

        new_log_alpha, new_alpha_opt, alpha_loss = gradient_step(
            alpha_loss_fn,
            state.log_alpha,
            state.alpha_opt_state,
            self.alpha_optimizer,
        )

        # 4. Polyak on critic targets
        new_target = optax.incremental_update(new_critic, state.target_critic_params, self.tau)

        new_state = SACState(
            actor_params=new_actor,
            critic_params=new_critic,
            log_alpha=new_log_alpha,
            actor_opt_state=new_actor_opt,
            critic_opt_state=new_critic_opt,
            alpha_opt_state=new_alpha_opt,
            target_critic_params=new_target,
        )
        return new_state, {
            "critic_loss": c_loss,
            "actor_loss": a_loss,
            "alpha_loss": alpha_loss,
            **critic_aux,
        }

    def act(
        self,
        state: SACState,
        obs: Float[Array, " d"],
        key: PRNGKeyArray,
    ) -> Array:
        """Sample an action from the current policy."""
        actor, head = self._reconstruct_actor(state.actor_params)
        features = actor(obs)
        dist = head(features)
        return dist.sample(key)

    # ------------------------------------------------------------------
    # Internal: partitioning and reconstruction
    # ------------------------------------------------------------------

    def _partition_params(self):
        """Extract actor, critic, and alpha param groups from the full module."""
        # Actor = actor MLP + action_head
        actor_module = (self.actor, self.action_head)
        actor_params = eqx.partition(actor_module, eqx.is_array)[0]

        # Critic = critic_1 + critic_2
        critic_module = (self.critic_1, self.critic_2)
        critic_params = eqx.partition(critic_module, eqx.is_array)[0]

        return actor_params, critic_params

    def _statics(self):
        """Return static (non-array) parts for actor and critic modules."""
        actor_module = (self.actor, self.action_head)
        actor_static = eqx.partition(actor_module, eqx.is_array)[1]

        critic_module = (self.critic_1, self.critic_2)
        critic_static = eqx.partition(critic_module, eqx.is_array)[1]

        return actor_static, critic_static

    def _reconstruct_actor(self, actor_params):
        """Combine actor params with static to get live actor + head."""
        actor_static = self._statics()[0]
        actor_module = eqx.combine(actor_params, actor_static)
        return actor_module[0], actor_module[1]

    def _reconstruct_critics(self, critic_params, critic_static):
        """Combine critic params with static to get live Q1, Q2."""
        critic_module = eqx.combine(critic_params, critic_static)
        return critic_module[0], critic_module[1]

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    def _critic_loss(
        self,
        critic_params,
        critic_static,
        target_critic_params,
        actor_params,
        actor_static,
        alpha,
        batch,
        key,
    ) -> tuple[Float[Array, ""], dict[str, Array]]:
        r"""Twin Q-network Bellman MSE loss.

        $$L_Q = \frac{1}{2B}\sum_i\bigl(Q_1(s,a) - y\bigr)^2
                + \bigl(Q_2(s,a) - y\bigr)^2$$

        where $y = r + \gamma(1-d)\bigl(\min(Q_1', Q_2')(s', a') - \alpha \log\pi(a'|s')\bigr)$.

        Returns:
            ``(loss, {"td_errors": abs_td_errors})`` for PER integration.
        """
        q1, q2 = self._reconstruct_critics(critic_params, critic_static)
        t_q1, t_q2 = self._reconstruct_critics(target_critic_params, critic_static)
        actor, head = eqx.combine(actor_params, actor_static)

        if self.discrete:
            # Q(s) -> (A,), index by action
            q1_sa = jax.vmap(q1)(batch.obs)
            q1_val = q1_sa[jnp.arange(q1_sa.shape[0]), batch.action.astype(jnp.int32)]
            q2_sa = jax.vmap(q2)(batch.obs)
            q2_val = q2_sa[jnp.arange(q2_sa.shape[0]), batch.action.astype(jnp.int32)]

            # Target: use all actions for expectation
            next_features = jax.vmap(actor)(batch.next_obs)
            next_dists = jax.vmap(head)(next_features)
            next_probs = next_dists.probs  # (B, A)
            next_log_probs = jnp.log(next_probs + LOG_EPS)

            t_q1_next = jax.vmap(t_q1)(batch.next_obs)  # (B, A)
            t_q2_next = jax.vmap(t_q2)(batch.next_obs)  # (B, A)
            min_q_next = jnp.minimum(t_q1_next, t_q2_next)  # (B, A)

            # V(s') = sum_a pi(a|s') * (min_Q(s',a) - alpha * log pi(a|s'))
            v_next = jnp.sum(next_probs * (min_q_next - alpha * next_log_probs), axis=-1)
        else:
            # Continuous: Q(s, a) -> scalar (squeeze (1,) -> ())
            q1_val = jax.vmap(lambda o, a: q1(jnp.concatenate([o, a])).squeeze())(batch.obs, batch.action)
            q2_val = jax.vmap(lambda o, a: q2(jnp.concatenate([o, a])).squeeze())(batch.obs, batch.action)

            # Sample next action from current policy using stable sample_and_log_prob
            next_features = jax.vmap(actor)(batch.next_obs)
            next_dists = jax.vmap(head)(next_features)
            next_actions, next_per_dim_lp = next_dists.sample_and_log_prob(key)
            next_log_probs = jnp.sum(next_per_dim_lp, axis=-1)

            t_q1_next = jax.vmap(lambda o, a: t_q1(jnp.concatenate([o, a])).squeeze())(batch.next_obs, next_actions)
            t_q2_next = jax.vmap(lambda o, a: t_q2(jnp.concatenate([o, a])).squeeze())(batch.next_obs, next_actions)
            min_q_next = jnp.minimum(t_q1_next, t_q2_next)
            v_next = min_q_next - alpha * next_log_probs

        done_mask = 1.0 - batch.done.astype(jnp.float32)
        td_target = batch.reward + self.gamma * done_mask * v_next

        td_errors = jnp.minimum(q1_val, q2_val) - td_target
        critic_loss = 0.5 * jnp.mean((q1_val - td_target) ** 2 + (q2_val - td_target) ** 2)
        return critic_loss, {"td_errors": jnp.abs(td_errors)}

    def _actor_loss(
        self,
        actor_params,
        actor_static,
        critic_params,
        critic_static,
        alpha,
        batch,
        key,
    ) -> Float[Array, ""]:
        r"""Policy loss: maximise expected Q minus entropy penalty.

        Continuous: $L_\pi = \frac{1}{B}\sum_i \alpha\log\pi(a|s) - \min(Q_1, Q_2)(s, a)$
        where $a \sim \pi(\cdot|s)$ via the reparameterisation trick.

        Discrete: $L_\pi = \frac{1}{B}\sum_i \sum_a \pi(a|s)\bigl(\alpha\log\pi(a|s) - \min(Q_1, Q_2)(s, a)\bigr)$
        """
        actor, head = eqx.combine(actor_params, actor_static)
        q1, q2 = self._reconstruct_critics(critic_params, critic_static)

        features = jax.vmap(actor)(batch.obs)
        dists = jax.vmap(head)(features)

        if self.discrete:
            probs = dists.probs  # (B, A)
            log_probs = jnp.log(probs + LOG_EPS)

            q1_all = jax.vmap(q1)(batch.obs)  # (B, A)
            q2_all = jax.vmap(q2)(batch.obs)  # (B, A)
            min_q = jnp.minimum(q1_all, q2_all)

            # Expectation over all actions
            actor_loss = jnp.mean(jnp.sum(probs * (alpha * log_probs - min_q), axis=-1))
        else:
            actions, per_dim_lp = dists.sample_and_log_prob(key)
            log_probs = jnp.sum(per_dim_lp, axis=-1)

            q1_val = jax.vmap(lambda o, a: q1(jnp.concatenate([o, a])).squeeze())(batch.obs, actions)
            q2_val = jax.vmap(lambda o, a: q2(jnp.concatenate([o, a])).squeeze())(batch.obs, actions)
            min_q = jnp.minimum(q1_val, q2_val)

            actor_loss = jnp.mean(alpha * log_probs - min_q)

        return actor_loss

    def _alpha_loss(
        self,
        log_alpha,
        actor_params,
        actor_static,
        batch,
        key,
    ) -> Float[Array, ""]:
        r"""Temperature loss for automatic entropy tuning.

        Continuous: $L_\alpha = -\alpha \cdot \mathrm{mean}(\log\pi(a|s) + H_{\mathrm{target}})$
        Discrete: $L_\alpha = -\alpha \cdot \mathrm{mean}(\sum_a \pi \cdot (\log\pi + H_t))$
        """
        alpha = jnp.exp(log_alpha)
        actor, head = eqx.combine(actor_params, actor_static)

        features = jax.vmap(actor)(batch.obs)
        dists = jax.vmap(head)(features)

        if self.discrete:
            probs = dists.probs  # (B, A)
            log_probs = jnp.log(probs + LOG_EPS)
            # Per-sample entropy contribution weighted by policy
            entropy_term = jnp.sum(probs * (log_probs + self.target_entropy), axis=-1)
            alpha_loss = -alpha * jnp.mean(entropy_term)
        else:
            actions, per_dim_lp = dists.sample_and_log_prob(key)
            log_probs = jnp.sum(per_dim_lp, axis=-1)
            alpha_loss = -alpha * jnp.mean(log_probs + self.target_entropy)

        return alpha_loss
