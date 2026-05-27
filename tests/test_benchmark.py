"""Forward-pass and per-step wall-clock benchmarks.

These tests measure compile time + training throughput for representative
agents. They use generous upper bounds so the assertion passes on any
reasonable machine; the value is the printed timing data, captured for
PR review and regression tracking.

``pytest.ini`` excludes ``@pytest.mark.benchmark`` from the default
run. Invoke explicitly:

    pytest tests/test_benchmark.py -m benchmark -s
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from rltrain.builders import agent as build_agent
from rltrain.builders import env as build_env
from rltrain.trainer import Trainer
from rltrain.transitions import make_transition


# Generous bounds — these prevent silent regressions from going unnoticed
# without false-positive flakes. The printed numbers carry the actual data.
COMPILE_BOUND_SEC = 60.0
WALL_CLOCK_BOUND_SEC = 60.0
WARM_CALL_BOUND_SEC = 5.0

N_STEPS = 1_000
HORIZON = 256
SEED = 0


@pytest.mark.benchmark
def test_ppo_compile_and_wall_clock(capsys) -> None:
    """Measure PPO ``learn`` compile time + 1000-step wall-clock on CartPole."""
    repo_root = Path(__file__).resolve().parent.parent
    agent_cfg = json.loads((repo_root / "examples" / "cartpole" / "ppo.json").read_text())
    env_cfg = json.loads((repo_root / "examples" / "cartpole" / "env.json").read_text())

    key = jax.random.key(SEED)
    k_agent, k_env, k_init, k_learn = jax.random.split(key, 4)
    agent = build_agent(**agent_cfg, key=k_agent)
    env = build_env(**env_cfg)

    # --- 1. Compile time: first call to agent.learn -------------------
    state = agent.init(k_init)
    env_state = env.reset(jax.random.split(k_env, 1)[0])
    obs_batch = jnp.broadcast_to(env_state.obs, (HORIZON, *env_state.obs.shape))
    batch = make_transition(
        obs=obs_batch,
        action=jnp.zeros(HORIZON, dtype=jnp.int32),
        reward=jnp.zeros(HORIZON),
        next_obs=obs_batch,
        done=jnp.zeros(HORIZON, dtype=jnp.bool_),
        log_prob=jnp.zeros(HORIZON),
        value=jnp.zeros(HORIZON),
    )

    learn_jit = eqx.filter_jit(agent.learn)

    t0 = time.perf_counter()
    new_state, _metrics = learn_jit(state, batch, k_learn)
    jax.block_until_ready(new_state.params)
    compile_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    new_state, _metrics = learn_jit(state, batch, k_learn)
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
    steps_per_sec = N_STEPS / wall_clock

    # Print under -s so PR reviewers can see the numbers.
    with capsys.disabled():
        print()
        print("PPO compile + wall-clock smoke benchmark")
        print("=" * 50)
        print(f"  compile_time:  {compile_time:7.3f} s   (first jitted learn call)")
        print(f"  warm_call:     {warm_call:7.3f} s   (second jitted learn call)")
        print(f"  wall_clock:    {wall_clock:7.3f} s   ({N_STEPS} env steps via Trainer)")
        print(f"  steps/sec:     {steps_per_sec:7.1f}")
        print("=" * 50)

    # Bounds are deliberately loose — they catch order-of-magnitude
    # regressions without flaking on machine variance.
    assert compile_time < COMPILE_BOUND_SEC, f"compile_time {compile_time:.2f}s > {COMPILE_BOUND_SEC}s"
    assert warm_call < WARM_CALL_BOUND_SEC, f"warm_call {warm_call:.2f}s > {WARM_CALL_BOUND_SEC}s"
    assert wall_clock < WALL_CLOCK_BOUND_SEC, f"wall_clock {wall_clock:.2f}s > {WALL_CLOCK_BOUND_SEC}s"
