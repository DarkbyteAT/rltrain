"""Smoke benchmark for PPO compile time and per-step wall-clock.

Loads the CartPole PPO config, JIT-compiles ``agent.learn``, and times:

1. First-call compile time (wall-clock of the first ``learn`` invocation).
2. Wall-clock for ``N_STEPS`` env steps via the Trainer.

Intended to be run before/after structural changes to ``learn`` (e.g.
the Python-loop → ``lax.scan`` refactor) so we can quote concrete numbers
in PR reviews. Not part of the test suite; not a microbenchmark — order
of magnitude is what matters.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from rltrain.builders import agent as build_agent
from rltrain.builders import env as build_env
from rltrain.trainer import Trainer


N_STEPS = 1_000
HORIZON = 256
SEED = 0


def main() -> None:
    """Run the PPO compile + wall-clock smoke benchmark."""
    repo_root = Path(__file__).resolve().parent.parent
    agent_cfg = json.loads((repo_root / "examples" / "cartpole" / "ppo.json").read_text())
    env_cfg = json.loads((repo_root / "examples" / "cartpole" / "env.json").read_text())

    key = jax.random.key(SEED)
    k_agent, k_env, k_init, k_learn = jax.random.split(key, 4)

    agent = build_agent(**agent_cfg, key=k_agent)
    env = build_env(**env_cfg)

    # --- 1. Compile time: first call to agent.learn -------------------
    state = agent.init(k_init)

    # Build a horizon-shaped batch from a quick rollout
    env_state = env.reset(jax.random.split(k_env, 1)[0])
    from rltrain.transitions import make_transition

    obs_batch = jnp.broadcast_to(env_state.obs, (HORIZON, *env_state.obs.shape))
    action_batch = jnp.zeros(HORIZON, dtype=jnp.int32)
    reward_batch = jnp.zeros(HORIZON)
    done_batch = jnp.zeros(HORIZON, dtype=jnp.bool_)
    batch = make_transition(
        obs=obs_batch,
        action=action_batch,
        reward=reward_batch,
        next_obs=obs_batch,
        done=done_batch,
        log_prob=jnp.zeros(HORIZON),
        value=jnp.zeros(HORIZON),
    )

    learn_jit = eqx.filter_jit(agent.learn)

    t0 = time.perf_counter()
    new_state, metrics = learn_jit(state, batch, k_learn)
    jax.block_until_ready(new_state.params)
    compile_time = time.perf_counter() - t0

    # Second call (warm) — for sanity, should be much faster
    t0 = time.perf_counter()
    new_state, metrics = learn_jit(state, batch, k_learn)
    jax.block_until_ready(new_state.params)
    warm_call = time.perf_counter() - t0

    # --- 2. Wall-clock: N_STEPS env steps via Trainer -----------------
    trainer = Trainer(
        agent,
        env,
        num_steps=N_STEPS,
        checkpoint_steps=N_STEPS,
        seed=SEED,
        callbacks=[],
    )
    t0 = time.perf_counter()
    _ = trainer.fit(jax.random.key(SEED))
    wall_clock = time.perf_counter() - t0

    print()
    print("PPO compile + wall-clock smoke benchmark")
    print("=" * 50)
    print(f"  compile_time:  {compile_time:7.3f} s   (first jitted learn call)")
    print(f"  warm_call:     {warm_call:7.3f} s   (second jitted learn call)")
    print(f"  wall_clock:    {wall_clock:7.3f} s   ({N_STEPS} env steps via Trainer)")
    print(f"  steps/sec:     {N_STEPS / wall_clock:7.1f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
