r"""Vanilla DQN agent — pure-functional JAX implementation.

Implements the Bellman MSE loss $L = \mathbb{E}[(r + \gamma \max_{a'} Q_{\theta^-}(s',a') \cdot (1 - d) - Q_\theta(s,a))^2]$
with epsilon-greedy exploration, target network Polyak averaging, and a fully
external training state (target params, optimizer state, epsilon).

The agent module holds architecture and static hyperparameters only.
Mutable training state lives outside as plain pytrees, keeping the
``eqx.partition``/``eqx.combine`` boundary clean.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray

from spike.networks import MLP
from spike.transitions import Transition


class VanillaDQN(eqx.Module):
    r"""Q-learning agent with a target network.

    The module stores the online Q-network and static hyperparameters.
    Target network parameters are maintained externally by the caller —
    this avoids doubling the pytree size and keeps partition/combine trivial.

    Args:
        obs_size: Dimensionality of the observation vector.
        num_actions: Number of discrete actions.
        width: Hidden layer width.
        depth: Number of hidden layers.
        gamma: Discount factor $\gamma \in [0, 1]$.
        key: PRNG key for weight initialisation.
    """

    q_net: MLP
    gamma: float = eqx.field(static=True)
    num_actions: int = eqx.field(static=True)

    def __init__(
        self,
        obs_size: int,
        num_actions: int,
        width: int,
        depth: int,
        gamma: float,
        *,
        key: PRNGKeyArray,
    ):
        self.q_net = MLP(in_size=obs_size, out_size=num_actions, width=width, depth=depth, key=key)
        self.gamma = gamma
        self.num_actions = num_actions

    def q_values(self, obs: Float[Array, " obs_dim"]) -> Float[Array, " num_actions"]:
        r"""Compute $Q_\theta(s, \cdot)$ for all actions."""
        return self.q_net(obs)

    def act(self, obs: Float[Array, " obs_dim"], key: PRNGKeyArray, epsilon: float) -> Array:
        r"""Epsilon-greedy action selection.

        With probability $\epsilon$ sample uniformly; otherwise $\arg\max_a Q(s,a)$.
        """
        q = self.q_values(obs)
        best_action = jnp.argmax(q)
        key_choice, key_rand = jax.random.split(key)
        random_action = jax.random.randint(key_rand, (), 0, self.num_actions)
        return jnp.where(jax.random.uniform(key_choice) < epsilon, random_action, best_action)

    def loss(self, target_params: "VanillaDQN", batch: Transition) -> Float[Array, ""]:
        r"""Bellman MSE loss.

        $$L = \frac{1}{B}\sum_i \bigl(r_i + \gamma \max_{a'} Q_{\theta^-}(s'_i, a') \cdot (1 - d_i) - Q_\theta(s_i, a_i)\bigr)^2$$

        Args:
            target_params: Dynamic parameters of the target network (combined
                with ``self``'s static structure via ``eqx.combine``).
            batch: Batched transition with leading dimension ``B``.
        """
        # Reconstruct the target network: target_params carries the learned
        # weights, while self provides the static module structure.
        _, static = eqx.partition(self, eqx.is_array)
        target_net = eqx.combine(target_params, static)

        # Vectorise Q-value computation over the batch
        q_all = jax.vmap(self.q_values)(batch.obs)  # (B, num_actions)
        q_sa = q_all[jnp.arange(q_all.shape[0]), batch.action.astype(jnp.int32)]  # (B,)

        target_q_all = jax.vmap(target_net.q_values)(batch.next_obs)  # (B, num_actions)
        target_max = jnp.max(target_q_all, axis=-1)  # (B,)

        td_target = batch.reward + self.gamma * target_max * (1.0 - batch.done.astype(jnp.float32))
        td_error = td_target - q_sa
        return jnp.mean(td_error**2)


def learn(
    params,
    static,
    target_params,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    batch: Transition,
    target_rate: float,
):
    r"""One DQN learning step: gradient descent + Polyak target update.

    Pure function suitable for ``jax.jit``. Computes the Bellman MSE loss,
    applies one optimizer step, and soft-updates the target network via
    ``optax.incremental_update`` with step size ``target_rate``.

    Args:
        params: Dynamic (array) leaves of the agent, from ``eqx.partition``.
        static: Static (non-array) leaves of the agent.
        target_params: Dynamic leaves of the target network.
        opt_state: Current optimizer state.
        optimizer: Optax gradient transformation.
        batch: Batched transitions to learn from.
        target_rate: Polyak averaging coefficient $\tau$; target is updated as
            $\theta^- \leftarrow \tau \theta + (1-\tau) \theta^-$.

    Returns:
        Tuple of ``(new_params, new_opt_state, new_target_params, metrics)``
        where ``metrics`` is a dict containing the scalar loss.
    """

    def loss_fn(p):
        a = eqx.combine(p, static)
        return a.loss(target_params, batch)

    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(params)

    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    new_target_params = optax.incremental_update(new_params, target_params, target_rate)

    return new_params, new_opt_state, new_target_params, {"loss": loss_val}
