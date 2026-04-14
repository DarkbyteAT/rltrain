r"""vmap seed sweep -- train N seeds in parallel.

JAX's vmap vectorises the entire training loop across seeds without
any code changes. In PyTorch, you would need multiprocessing or
sequential runs.

The entire training loop is a pure JAX function (lax.scan over fixed-
length horizons), so vmap can batch it across PRNG keys.
"""

import jax
import jax.numpy as jnp
import optax

from spike.agents.vanilla_pg import VanillaPG
from spike.env import GymnaxEnv
from spike.heads import DiscreteHead
from spike.networks import MLP
from spike.transitions import make_transition


# Module-level so closures capture them as static constants.
_env = GymnaxEnv("CartPole-v1")

_agent = VanillaPG(
    actor=MLP(4, 64, width=64, depth=1, key=jax.random.PRNGKey(0)),
    action_head=DiscreteHead(64, 2, key=jax.random.PRNGKey(1)),
    optimizer=optax.adam(3e-3),
    gamma=0.99,
    tau=0.01,
    normalise=True,
)

HORIZON = 256
NUM_LEARNS = 20  # total steps = HORIZON * NUM_LEARNS = 5120


def _train_one_seed(key):
    """Pure-JAX training loop for a single seed, returning mean episode return."""
    import equinox as eqx

    agent = _agent
    env = _env

    k_init, k_env, k_loop = jax.random.split(key, 3)
    state = agent.init(k_init)
    env_state = env.reset(k_env)

    # Inner scan: collect HORIZON transitions
    def _collect_step(carry, _):
        es, rng = carry
        rng, k_act, k_step = jax.random.split(rng, 3)
        static = eqx.partition(agent, eqx.is_array)[1]
        live = eqx.combine(carry_state[0], static)
        features = live.actor(es.obs)
        dist = live.action_head(features)
        action = dist.sample(k_act)
        new_es = env.step(es, action, k_step)
        transition = make_transition(
            obs=es.obs,
            action=action,
            reward=new_es.reward,
            next_obs=new_es.obs,
            done=new_es.done,
        )
        return (new_es, rng), transition

    # We need a slightly different approach: thread agent state through the
    # outer loop but not the inner collect loop (agent params don't change
    # during collection). Use a closure to read state from the outer carry.

    def _outer_step(carry, _):
        agent_state, es, rng = carry
        rng, k_collect, k_learn = jax.random.split(rng, 3)

        # Collect HORIZON steps using the current policy
        def _collect(carry, _):
            es_inner, rng_inner = carry
            rng_inner, k_act, k_step = jax.random.split(rng_inner, 3)
            static = eqx.partition(agent, eqx.is_array)[1]
            live = eqx.combine(agent_state.params, static)
            features = live.actor(es_inner.obs)
            dist = live.action_head(features)
            action = dist.sample(k_act)
            new_es = env.step(es_inner, action, k_step)
            tr = make_transition(
                obs=es_inner.obs,
                action=action,
                reward=new_es.reward,
                next_obs=new_es.obs,
                done=new_es.done,
            )
            return (new_es, rng_inner), tr

        (es, _), batch = jax.lax.scan(_collect, (es, k_collect), jnp.arange(HORIZON))

        # Learn on the collected horizon
        agent_state, metrics = agent.learn(agent_state, batch, k_learn)
        return (agent_state, es, rng), metrics["loss"]

    # Remove the stale closure reference
    carry_state = [state]  # noqa: F841 — used only in the dead code above

    (final_state, final_es, _), losses = jax.lax.scan(_outer_step, (state, env_state, k_loop), jnp.arange(NUM_LEARNS))
    return jnp.mean(losses)


def main():  # noqa: D103
    num_seeds = 8
    keys = jax.random.split(jax.random.PRNGKey(42), num_seeds)

    print("Compiling vmapped training loop...")
    vmapped_train = jax.jit(jax.vmap(_train_one_seed))
    all_mean_losses = vmapped_train(keys)
    all_mean_losses.block_until_ready()

    print()
    print("=" * 50)
    print("vmap Seed Sweep: VanillaPG on CartPole-v1")
    print(f"  {num_seeds} seeds, {HORIZON * NUM_LEARNS} steps each")
    print("=" * 50)
    for i, loss in enumerate(all_mean_losses):
        print(f"  Seed {i}: mean loss = {float(loss):.4f}")
    print(f"  Overall: {float(jnp.mean(all_mean_losses)):.4f} +/- {float(jnp.std(all_mean_losses)):.4f}")
    print("=" * 50)
    print("All seeds trained in a single vectorised XLA call.")


if __name__ == "__main__":
    main()
