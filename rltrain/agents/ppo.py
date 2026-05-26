r"""Proximal Policy Optimisation (PPO) — extends AdvantageAC with clipped surrogate.

The clipped surrogate objective is

$$\mathcal{L}^{\mathrm{CLIP}}(\theta) = -\mathbb{E}\!\bigl[
    \min\bigl(r_t(\theta)\,A_t,\;
    \mathrm{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\,A_t\bigr)
\bigr]$$

where $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{\mathrm{old}}}(a_t|s_t)$
is the probability ratio.  Mini-batch epochs over the horizon buffer allow
multiple gradient steps per data collection phase.  Composable
:class:`EpochTerminator` instances can short-circuit the remaining epochs
once a stopping condition (e.g. KL divergence) is met.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from rltrain.agents.agent import OnPolicyAgent, TrainState, gradient_step
from rltrain.agents.ppo_terminators import EpochTerminator
from rltrain.heads import Head
from rltrain.math import center, gae
from rltrain.transitions import Transition


def _bake_terminator_chain(
    terminators: tuple[EpochTerminator, ...],
) -> Callable[[dict[str, Float[Array, ""]]], Bool[Array, ""]]:
    """Collapse a static tuple of terminators into a single callable.

    The closure iterates the tuple at trace time, but the scan body sees one
    opaque function — so the traced graph is independent of the tuple's
    contents. Adding or removing terminators only re-traces by changing this
    closure, not by changing the structure of the scan body.
    """

    def check_stop(metrics: dict[str, Float[Array, ""]]) -> Bool[Array, ""]:
        stopped = jnp.array(False)
        for t in terminators:
            stopped = jnp.logical_or(stopped, t.should_stop(metrics))
        return stopped

    return check_stop


class PPO(OnPolicyAgent):
    r"""PPO agent with clipped surrogate objective, mini-batch epochs, and optional terminators.

    Inherits ``init`` and ``act`` from :class:`OnPolicyAgent`. Overrides ``learn``
    for the multi-epoch mini-batch loop with composable
    :class:`~rltrain.agents.ppo_terminators.EpochTerminator` short-circuiting.

    Attributes:
        critic: Value-function network. Any ``eqx.Module`` mapping
            ``obs -> [1]``.
        gamma: Discount factor $\gamma \in [0, 1]$.
        tau: Entropy regularisation coefficient.
        beta_critic: Weight on the critic MSE loss term.
        lambda_gae: GAE smoothing parameter $\lambda \in [0, 1]$.
        eps_clip: Clipping parameter $\varepsilon$ for the surrogate ratio.
        num_epochs: Number of optimisation epochs per horizon.
        minibatch_size: Mini-batch size within each epoch.
        epoch_terminators: Tuple of
            :class:`~rltrain.agents.ppo_terminators.EpochTerminator` instances
            evaluated at the end of each epoch. When any returns ``True``,
            remaining epochs are masked out (parameters frozen, loss not
            accumulated). Empty tuple disables early stopping.
    """

    critic: eqx.Module
    gamma: float = eqx.field(static=True)
    tau: float = eqx.field(static=True)
    beta_critic: float = eqx.field(static=True)
    lambda_gae: float = eqx.field(static=True)
    eps_clip: float = eqx.field(static=True)
    num_epochs: int = eqx.field(static=True)
    minibatch_size: int = eqx.field(static=True)
    epoch_terminators: tuple[EpochTerminator, ...] = eqx.field(static=True, default=())
    # Closure baked at construction time over ``epoch_terminators``. The scan
    # body calls this single opaque callable instead of iterating the static
    # tuple inline, so the traced graph is independent of the tuple's contents.
    _check_stop: Callable[[dict[str, Float[Array, ""]]], Bool[Array, ""]] = eqx.field(static=True, init=False)

    def __init__(
        self,
        actor: eqx.Module,
        action_head: Head,
        optimizer: optax.GradientTransformation,
        critic: eqx.Module,
        gamma: float,
        tau: float,
        beta_critic: float,
        lambda_gae: float,
        eps_clip: float,
        num_epochs: int,
        minibatch_size: int,
        epoch_terminators: tuple[EpochTerminator, ...] = (),
    ):
        """Initialise PPO and bake the terminator chain into a single callable."""
        self.actor = actor
        self.action_head = action_head
        self.optimizer = optimizer
        self.critic = critic
        self.gamma = gamma
        self.tau = tau
        self.beta_critic = beta_critic
        self.lambda_gae = lambda_gae
        self.eps_clip = eps_clip
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.epoch_terminators = tuple(epoch_terminators)
        self._check_stop = _bake_terminator_chain(self.epoch_terminators)

    # --------------- Protocol methods ---------------

    def learn(
        self, state: TrainState, batch: Transition, key: PRNGKeyArray
    ) -> tuple[TrainState, dict[str, Float[Array, ""]]]:
        r"""Run a PPO update on one horizon of transitions.

        Computes GAE advantages from the frozen old policy, then runs a
        double ``lax.scan`` — outer over epochs, inner over mini-batches.
        After each epoch, the mean approximate KL across the epoch's
        mini-batches is computed and passed to the baked
        :attr:`_check_stop` chain; once it fires, a JAX boolean ``stopped``
        flag carried through the scan masks all subsequent parameter,
        optimiser-state, and loss updates, so the trace shape stays static
        while early-stop semantics are honoured at runtime.

        Args:
            state: Current :class:`TrainState`.
            batch: Horizon-length :class:`Transition` collected with the old
                policy. Leading axis is the horizon dimension.
            key: PRNG key consumed for per-epoch mini-batch shuffles.

        Returns:
            A tuple of:

            - The updated :class:`TrainState` (frozen from the point a
              terminator fired, if any).
            - A metrics dict containing the scalar keys ``"loss"`` (mean
              surrogate-plus-critic-plus-entropy loss averaged over executed
              mini-batches) and ``"approx_kl"`` (mean
              ``log_pi_old - log_pi_new`` over the last executed epoch).
        """
        static = eqx.partition(self, eqx.is_array)[1]

        # 1. Old log-probs (frozen reference policy).
        agent_old = eqx.combine(state.params, static)
        old_features = jax.vmap(agent_old.actor)(batch.obs)
        old_dists = jax.vmap(agent_old.action_head)(old_features)
        old_log_probs = jax.lax.stop_gradient(old_dists.log_prob(batch.action))

        # 2. Values + GAE.
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
        advantages = jax.lax.stop_gradient(center(advantages))
        returns = jax.lax.stop_gradient(returns)

        horizon_size = batch.obs.shape[0]
        num_minibatches = horizon_size // self.minibatch_size
        total = num_minibatches * self.minibatch_size

        def _minibatch_body(mb_carry, xs):
            params, opt_state, total_loss, kl_sum, stopped = mb_carry
            mb, mb_old_lp, mb_a, mb_r = xs

            def loss_fn(p):
                return eqx.combine(p, static)._ppo_loss(mb, mb_old_lp, mb_a, mb_r)

            new_params, new_opt_state, loss_val = gradient_step(loss_fn, params, opt_state, self.optimizer)

            # Approximate KL on this minibatch under the post-step policy.
            new_agent = eqx.combine(new_params, static)
            new_features = jax.vmap(new_agent.actor)(mb.obs)
            new_dists = jax.vmap(new_agent.action_head)(new_features)
            new_log_probs = new_dists.log_prob(mb.action)
            mb_kl = jnp.mean(mb_old_lp - new_log_probs)

            # Freeze parameters once a previous epoch fired a terminator.
            params = jax.tree.map(lambda old, new: jnp.where(stopped, old, new), params, new_params)
            opt_state = jax.tree.map(lambda old, new: jnp.where(stopped, old, new), opt_state, new_opt_state)
            total_loss = jnp.where(stopped, total_loss, total_loss + loss_val)
            kl_sum = kl_sum + mb_kl

            return (params, opt_state, total_loss, kl_sum, stopped), None

        def _epoch_body(epoch_carry, _epoch_idx):
            params, opt_state, total_loss, approx_kl_last, stopped, key = epoch_carry
            key, epoch_key = jax.random.split(key)

            # Shuffle then reshape into (num_minibatches, minibatch_size, ...)
            # so lax.scan iterates the leading axis.
            perm = jax.random.permutation(epoch_key, horizon_size)[:total]

            def _shuffle_and_reshape(x):
                return x[perm].reshape(num_minibatches, self.minibatch_size, *x.shape[1:])

            mb_batch = jax.tree.map(_shuffle_and_reshape, batch)
            mb_olp = _shuffle_and_reshape(old_log_probs)
            mb_adv = _shuffle_and_reshape(advantages)
            mb_ret = _shuffle_and_reshape(returns)

            init_inner = (params, opt_state, total_loss, jnp.zeros(()), stopped)
            (params, opt_state, total_loss, kl_sum, _), _ = jax.lax.scan(
                _minibatch_body, init_inner, (mb_batch, mb_olp, mb_adv, mb_ret)
            )

            epoch_kl_mean = kl_sum / jnp.maximum(num_minibatches, 1)
            approx_kl_last = jnp.where(stopped, approx_kl_last, epoch_kl_mean)
            new_stopped = jnp.logical_or(stopped, self._check_stop({"approx_kl": epoch_kl_mean}))

            return (params, opt_state, total_loss, approx_kl_last, new_stopped, key), None

        init_outer = (
            state.params,
            state.opt_state,
            jnp.zeros(()),  # total_loss
            jnp.zeros(()),  # approx_kl_last
            jnp.array(False),
            key,
        )
        (params, opt_state, total_loss, approx_kl_last, _stopped, _key), _ = jax.lax.scan(
            _epoch_body, init_outer, jnp.arange(self.num_epochs)
        )

        new_state = TrainState(
            params=params,
            opt_state=opt_state,
            target_params=state.target_params,
        )
        n_steps = jnp.array(self.num_epochs * num_minibatches, dtype=jnp.float32)
        return new_state, {
            "loss": total_loss / jnp.maximum(n_steps, 1.0),
            "approx_kl": approx_kl_last,
        }

    # --------------- Internal ---------------

    def _ppo_loss(
        self,
        transitions: Transition,
        old_log_probs: Float[Array, " T"],
        advantages: Float[Array, " T"],
        returns: Float[Array, " T"],
    ) -> Float[Array, ""]:
        r"""Clipped surrogate loss for a single mini-batch.

        $$\mathcal{L} = -\frac{1}{T}\sum_t \min(r_t A_t, \mathrm{clip}(r_t) A_t)
          + \beta_c\,(R_t - V(s_t))^2 - \tau\,H[\pi]$$

        Args:
            transitions: Mini-batch of transitions (length ``T``).
            old_log_probs: ``log \pi_{old}(a_t | s_t)`` from the frozen
                reference policy.
            advantages: Per-step centred advantages $\hat A_t$.
            returns: Per-step targets for the critic.

        Returns:
            The scalar loss.
        """
        features = jax.vmap(self.actor)(transitions.obs)
        dists = jax.vmap(self.action_head)(features)
        log_probs = dists.log_prob(transitions.action)
        entropy = dists.entropy()

        ratio = jnp.exp(log_probs - old_log_probs)
        clipped_ratio = jnp.clip(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)
        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        actor_loss = -jnp.mean(jnp.minimum(surr1, surr2))

        values = jax.vmap(lambda o: self.critic(o).squeeze(-1))(transitions.obs)
        critic_loss = jnp.mean((returns - values) ** 2)

        entropy_loss = -self.tau * jnp.mean(entropy)

        return actor_loss + self.beta_critic * critic_loss + entropy_loss
