r"""Scan benchmark -- wall-clock timing across env strategies.

Proves that lax.scan fuses the training loop into a single XLA kernel,
delivering significant speedup over Python-level loops.

Compares:
- Scan strategy (GymnaxEnv, lax.scan inner loop)
- Python-loop strategy (GymnaxEnv, forced Python outer loop)

The first call includes JIT compilation; the second is pure execution.
"""

import time

import jax
import optax

from rltrain.agents.ppo import PPO
from rltrain.env import GymnaxEnv
from rltrain.heads import DiscreteHead
from rltrain.networks import MLP
from rltrain.trainer import PythonLoop, ScanLoop, Trainer


def _make_agent(key):
    k1, k2, k3 = jax.random.split(key, 3)
    return PPO(
        actor=MLP(4, 64, width_size=64, depth=1, key=k1),
        action_head=DiscreteHead(64, 2, key=k2),
        critic=MLP(4, 1, width_size=64, depth=1, key=k3),
        optimizer=optax.adam(3e-3),
        gamma=0.99,
        tau=0.01,
        beta_critic=0.5,
        lambda_gae=0.95,
        eps_clip=0.2,
        num_epochs=4,
        minibatch_size=64,
    )


def _time_strategy(agent, env, num_steps, key, *, use_scan):
    """Time a training run, forcing scan or Python-loop strategy."""
    loop = ScanLoop() if use_scan else PythonLoop()
    trainer = Trainer(agent, env, num_steps=num_steps, checkpoint_steps=num_steps, loop=loop)

    # Warm-up: first call compiles XLA kernels
    k1, k2 = jax.random.split(key)
    start = time.perf_counter()
    _ = trainer.fit(k1)
    t_compile = time.perf_counter() - start

    # Timed run: pure execution, no compilation
    start = time.perf_counter()
    _ = trainer.fit(k2)
    t_execute = time.perf_counter() - start

    return t_compile, t_execute


def main():  # noqa: D103
    key = jax.random.PRNGKey(0)
    num_steps = 5120

    agent = _make_agent(key)
    env = GymnaxEnv("CartPole-v1")

    k_scan, k_python = jax.random.split(key)
    compile_scan, exec_scan = _time_strategy(agent, env, num_steps, k_scan, use_scan=True)
    compile_py, exec_py = _time_strategy(agent, env, num_steps, k_python, use_scan=False)

    speedup = exec_py / exec_scan if exec_scan > 0 else float("inf")

    print("=" * 55)
    print("Scan Benchmark: PPO on CartPole-v1")
    print("=" * 55)
    print(f"Steps:            {num_steps}")
    print()
    print("Strategy          Compile+Exec     Execute only")
    print(f"Scan (lax.scan)   {compile_scan:>8.3f}s        {exec_scan:>8.4f}s")
    print(f"Python loop       {compile_py:>8.3f}s        {exec_py:>8.4f}s")
    print()
    print(f"Scan speedup (execute only): {speedup:.1f}x")
    print("=" * 55)


if __name__ == "__main__":
    main()
