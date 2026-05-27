r"""JIT trace -- inspect the compiled program and measure compilation time.

Prints the JAX program representation (jaxpr) for one learn step and
compares first-call (compile + execute) vs subsequent (execute only) timing.
"""

import time

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
    k1, k2, k_data = jax.random.split(key, 3)

    agent = VanillaPG(
        actor=MLP(4, 32, width=32, depth=1, key=k1),
        action_head=DiscreteHead(32, 2, key=k2),
        optimizer=optax.adam(1e-3),
        gamma=0.99,
        tau=0.01,
        normalise=False,
    )

    state = agent.init(key)
    batch = _make_batch(k_data)

    # ---- 1. Trace the jaxpr for learn() ----
    _params, static = eqx.partition(agent, eqx.is_array)

    def learn_from_params(p):
        s = TrainState(params=p, opt_state=state.opt_state, target_params=state.target_params)
        new_s, metrics = agent.learn(s, batch, jax.random.PRNGKey(0))
        return metrics["loss"]

    jaxpr = jax.make_jaxpr(learn_from_params)(state.params)
    jaxpr_str = str(jaxpr)
    num_eqns = len(jaxpr.jaxpr.eqns)

    print("=" * 55)
    print("JIT Trace: VanillaPG learn() Inspection")
    print("=" * 55)
    print(f"JAXPR size:     {len(jaxpr_str):,} characters")
    print(f"JAXPR equations: {num_eqns}")
    print()

    # ---- 2. Time compilation vs execution ----
    learn_jit = eqx.filter_jit(agent.learn)
    k_learn = jax.random.PRNGKey(0)

    # First call: includes XLA compilation
    start = time.perf_counter()
    result = learn_jit(state, batch, k_learn)
    # Block until computation finishes (JAX is async)
    jax.tree.map(lambda x: x.block_until_ready(), result)
    t_compile = time.perf_counter() - start

    # Subsequent calls: pure execution
    n_iters = 100
    start = time.perf_counter()
    for _ in range(n_iters):
        result = learn_jit(state, batch, k_learn)
    jax.tree.map(lambda x: x.block_until_ready(), result)
    t_execute = (time.perf_counter() - start) / n_iters

    speedup = t_compile / t_execute if t_execute > 0 else float("inf")

    print(f"First call (compile + exec): {t_compile:.4f}s")
    print(f"Subsequent calls (exec):     {t_execute:.6f}s")
    print(f"Compile overhead:            {speedup:.0f}x")
    print()
    print("First 5 JAXPR equations:")
    for eqn in jaxpr.jaxpr.eqns[:5]:
        print(f"  {eqn.primitive.name}  ({len(eqn.invars)} in, {len(eqn.outvars)} out)")
    print("  ...")
    print("=" * 55)


if __name__ == "__main__":
    main()
