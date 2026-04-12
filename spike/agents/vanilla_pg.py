r"""Vanilla policy gradient (REINFORCE) as a pure-functional JAX agent.

The REINFORCE loss is

$$\mathcal{L}(\theta) = -\mathbb{E}\!\bigl[\log \pi_\theta(a|s)\,G_t\bigr]
                        - \tau\,\mathbb{E}\!\bigl[H[\pi_\theta(\cdot|s)]\bigr]$$

where $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$ are the discounted returns
and $\tau$ controls the entropy bonus.

The agent module is **static** — it holds network architecture, hyperparameters,
and the optimizer.  All trainable state lives in a ``TrainState`` pytree.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray

from spike.agents.agent import TrainState, gradient_step, zero_target_params
from spike.heads import DiscreteHead
from spike.networks import MLP
from spike.transitions import Transition


# ---------------------------------------------------------------------------
# Discounted returns via reverse lax.scan
# ---------------------------------------------------------------------------


def discount(rewards: Float[Array, " T"], dones: Float[Array, " T"], gamma: float) -> Float[Array, " T"]:
    r"""Compute discounted returns $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$.

    Uses ``jax.lax.scan`` in reverse so the operation is jittable.
    Episode boundaries (``dones == 1``) reset the accumulator.
    """

    def _step(
        acc: Float[Array, ""], xs: tuple[Float[Array, ""], Float[Array, ""]]
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        r, d = xs
        acc = r + gamma * acc * (1.0 - d)
        return acc, acc

    _, returns = jax.lax.scan(
        _step,
        jnp.zeros(()),
        (rewards, dones),
        reverse=True,
    )
    return returns


# ---------------------------------------------------------------------------
# Agent module (static — no trainable array leaves)
# ---------------------------------------------------------------------------


class VanillaPG(eqx.Module):
    r"""Vanilla policy gradient agent as a static Equinox module.

    The module holds network architecture, hyperparameters, and the optimizer.
    Trainable parameters live externally in a ``TrainState`` pytree, which
    is threaded through ``learn`` and ``act`` as an explicit argument.

    This separation means JIT traces the agent once (as a compile-time
    constant) and ``jax.grad`` composes through ``learn`` because the
    state is a pure pytree.
    """

    # Network architecture (used to reconstruct model from params)
    actor: MLP
    action_head: DiscreteHead

    # Optimizer (static — no array leaves)
    optimizer: optax.GradientTransformation = eqx.field(static=True)

    # Hyperparameters (all static)
    gamma: float = eqx.field(static=True)
    tau: float = eqx.field(static=True)
    normalise: bool = eqx.field(static=True)

    # --------------- Protocol methods ---------------

    def init(self, key: PRNGKeyArray) -> TrainState:
        """Construct the initial training state.

        Partitions the model into trainable params and static structure,
        initialises the optimizer, and creates zero-sentinel target params
        (unused by on-policy agents, but required for uniform TrainState).
        """
        params, _static = eqx.partition(self, eqx.is_array)
        opt_state = self.optimizer.init(params)
        target_params = zero_target_params(params)
        return TrainState(params=params, opt_state=opt_state, target_params=target_params)

    def learn(self, state: TrainState, batch: Transition) -> tuple[TrainState, dict[str, Float[Array, ""]]]:
        r"""One gradient step on the REINFORCE loss.

        Args:
            state: Current training state (params, opt_state, target_params).
            batch: Episode transitions to compute the loss over.

        Returns:
            ``(new_state, metrics)`` where target_params are unchanged
            (on-policy agents do not use target networks).
        """
        static = eqx.partition(self, eqx.is_array)[1]

        def loss_fn(params):
            agent = eqx.combine(params, static)
            return agent._loss(batch)

        new_params, new_opt_state, loss_val = gradient_step(loss_fn, state.params, state.opt_state, self.optimizer)
        new_state = TrainState(
            params=new_params,
            opt_state=new_opt_state,
            target_params=state.target_params,
        )
        return new_state, {"loss": loss_val}

    def act(self, state: TrainState, obs: Float[Array, " d"], key: PRNGKeyArray) -> Array:
        """Sample an action from the policy.

        Reconstructs the model from ``state.params`` and the agent's static
        structure, then forward-passes through actor + action head.
        """
        static = eqx.partition(self, eqx.is_array)[1]
        agent = eqx.combine(state.params, static)
        features = agent.actor(obs)
        dist = agent.action_head(features)
        return dist.sample(key)

    # --------------- Internal ---------------

    def _loss(self, transitions: Transition) -> Float[Array, ""]:
        r"""REINFORCE loss over a batch of transitions.

        $$-\frac{1}{T}\sum_t \log\pi(a_t|s_t)\,G_t
          \;-\;\tau\,\frac{1}{T}\sum_t H[\pi(\cdot|s_t)]$$

        Called on a reconstructed agent (with params combined), so
        ``self.actor`` and ``self.action_head`` carry trainable weights.
        """
        returns = discount(transitions.reward, transitions.done.astype(jnp.float32), self.gamma)

        if self.normalise:
            returns = (returns - jnp.mean(returns)) / (jnp.std(returns) + 1e-8)

        features = jax.vmap(self.actor)(transitions.obs)
        dists = jax.vmap(self.action_head)(features)

        log_probs = dists.log_prob(transitions.action.squeeze(-1))
        entropy = dists.entropy()

        actor_loss = -jnp.mean(log_probs * returns)
        entropy_loss = -self.tau * jnp.mean(entropy)
        return actor_loss + entropy_loss
