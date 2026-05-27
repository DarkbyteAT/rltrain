r"""MAML composability -- second-order gradients through agent.learn().

Proves that jax.grad composes natively through the full learn() pipeline.
The meta-gradient (through an inner gradient step) differs from the direct
gradient, confirming that second-order information flows correctly.

In PyTorch, this requires create_graph=True and careful manual management.
In JAX, it works out of the box because learn() is a pure function.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from rltrain.agents.agent import TrainState
from rltrain.agents.vanilla_pg import VanillaPG
from rltrain.heads import DiscreteHead
from rltrain.networks import MLP
from rltrain.transitions import make_transition


def _make_batch(key, horizon=64):
    """Synthesise a batch of fake CartPole transitions."""
    k1, k2, k3 = jax.random.split(key, 3)
    return make_transition(
        obs=jax.random.normal(k1, (horizon, 4)),
        action=jax.random.randint(k2, (horizon,), 0, 2),
        reward=jax.random.uniform(k3, (horizon,)),
        next_obs=jax.random.normal(k1, (horizon, 4)),
        done=jnp.zeros(horizon, dtype=jnp.bool_),
        log_prob=jnp.zeros(horizon),
        value=jnp.zeros(horizon),
    )


def main():  # noqa: D103
    key = jax.random.PRNGKey(0)
    k1, k2, k_data1, k_data2 = jax.random.split(key, 4)

    agent = VanillaPG(
        actor=MLP(4, 32, width=32, depth=1, key=k1),
        action_head=DiscreteHead(32, 2, key=k2),
        optimizer=optax.adam(1e-3),
        gamma=0.99,
        tau=0.01,
        normalise=False,
    )

    state = agent.init(key)
    _params, static = eqx.partition(agent, eqx.is_array)

    train_batch = _make_batch(k_data1)
    eval_batch = _make_batch(k_data2)

    # ---- Direct gradient: grad of loss w.r.t. params (first-order) ----
    def direct_loss(params):
        a = eqx.combine(params, static)
        return a._loss(eval_batch)

    direct_grad = jax.grad(direct_loss)(state.params)

    # ---- Meta-gradient: grad through one inner gradient step (second-order) ----
    def meta_loss(init_params):
        # Inner loop: one gradient step on the train batch
        s = TrainState(
            params=init_params,
            opt_state=state.opt_state,
            target_params=state.target_params,
        )
        updated_state, _metrics = agent.learn(s, train_batch, jax.random.PRNGKey(99))

        # Outer loss: evaluate the updated params on the eval batch
        updated_agent = eqx.combine(updated_state.params, static)
        return updated_agent._loss(eval_batch)

    meta_grad = jax.grad(meta_loss)(state.params)

    # ---- Compare ----
    meta_leaves = jax.tree.leaves(meta_grad)
    direct_leaves = jax.tree.leaves(direct_grad)

    diffs = [float(jnp.max(jnp.abs(m - d))) for m, d in zip(meta_leaves, direct_leaves, strict=False)]
    gradients_differ = any(d > 1e-6 for d in diffs)

    print("=" * 55)
    print("MAML Composability: Second-Order Gradients")
    print("=" * 55)
    print("Agent:               VanillaPG (4 -> 32 -> 2)")
    print(f"Gradient leaves:     {len(meta_leaves)}")
    print("Max |meta - direct| per leaf:")
    for i, d in enumerate(diffs):
        print(f"  leaf {i}: {d:.6f}")
    print()
    print(f"Gradients differ:    {gradients_differ}")
    assert gradients_differ, "Meta and direct gradients should differ!"
    print("PASS: jax.grad composes through agent.learn()")
    print("      -- second-order gradients work natively.")
    print("=" * 55)


if __name__ == "__main__":
    main()
