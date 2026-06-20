"""Tests for ``MultiSeedTrainer`` — vmap-over-seeds parallelism."""

from __future__ import annotations

import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from rltrain.agents.ppo import PPO
from rltrain.agents.vanilla_pg import VanillaPG
from rltrain.env import GymnaxEnv
from rltrain.heads import DiscreteHead
from rltrain.networks import MLP
from rltrain.trainer import MultiSeedTrainer
from rltrain.trainer._trainer import Trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OBS_DIM = 4
NUM_ACTIONS = 2
HIDDEN = 32


def _make_pg_factory():
    """Returns an agent_factory closure that builds a tiny VanillaPG per key."""

    def factory(key: jax.Array) -> VanillaPG:
        k1, k2 = jax.random.split(key)
        return VanillaPG(
            actor=MLP(OBS_DIM, HIDDEN, width_size=HIDDEN, depth=1, key=k1),
            action_head=DiscreteHead(HIDDEN, NUM_ACTIONS, key=k2),
            optimizer=optax.adam(1e-3),
            gamma=0.99,
            tau=0.01,
            normalise=True,
        )

    return factory


def _make_ppo_factory(hidden: int = HIDDEN):
    """Returns an agent_factory closure that builds a tiny PPO per key."""

    def factory(key: jax.Array) -> PPO:
        k1, k2, k3 = jax.random.split(key, 3)
        return PPO(
            actor=MLP(OBS_DIM, hidden, width_size=hidden, depth=1, key=k1),
            action_head=DiscreteHead(hidden, NUM_ACTIONS, key=k2),
            critic=MLP(OBS_DIM, 1, width_size=hidden, depth=1, key=k3),
            optimizer=optax.adam(3e-4),
            gamma=0.99,
            tau=0.01,
            beta_critic=0.5,
            lambda_gae=0.95,
            eps_clip=0.2,
            num_epochs=2,
            minibatch_size=32,
        )

    return factory


# ---------------------------------------------------------------------------
# Unit: factory invocation + key splitting
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_factory_traced_under_vmap_produces_distinct_per_seed_params():
    """The agent factory is vmapped over ``n_seeds`` distinct keys.

    ``eqx.filter_vmap`` traces the factory body once with a vectorised
    tracer (Python-side call count is 1), so the contract we actually
    care about is: the resulting stacked agent has ``n_seeds`` distinct
    per-seed parameter slices — i.e. the keys flowing through the trace
    differ along the leading axis.
    """
    # Given: a trainer with n_seeds=4
    env = GymnaxEnv("CartPole-v1")
    n_seeds = 4
    trainer = MultiSeedTrainer(
        _make_pg_factory(),
        env,
        num_steps=256,
        n_seeds=n_seeds,
        checkpoint_steps=256,
        batch_size=32,
    )

    # When: we build the initial state
    agent_stack, _carry = trainer.make_initial_state(jax.random.key(7))

    # Then: every pair of per-seed parameter slices differs (distinct keys
    # produced distinct params for each linear layer).
    arrays, _static = eqx.partition(agent_stack, eqx.is_array)
    weight_leaves = [leaf for leaf in jax.tree.leaves(arrays) if leaf.ndim >= 2]
    assert weight_leaves, "Expected at least one weight matrix leaf"
    for leaf in weight_leaves:
        for i in range(n_seeds):
            for j in range(i + 1, n_seeds):
                assert not jnp.allclose(leaf[i], leaf[j]), (
                    f"Per-seed slices of leaf shape {leaf.shape} collide at seeds {i},{j}"
                )


@pytest.mark.unit
def test_key_splitting_deterministic():
    """Calling ``make_initial_state`` twice with the same master key gives identical sub-keys."""
    # Given: a trainer
    env = GymnaxEnv("CartPole-v1")
    trainer = MultiSeedTrainer(
        _make_pg_factory(),
        env,
        num_steps=256,
        n_seeds=3,
        checkpoint_steps=256,
        batch_size=32,
    )

    # When: we split twice with the same master key
    bk1, ek1, fk1 = trainer._split_master_key(jax.random.key(99))
    bk2, ek2, fk2 = trainer._split_master_key(jax.random.key(99))

    # Then: each component is identical, and the three components are distinct
    assert jnp.array_equal(jax.random.key_data(bk1), jax.random.key_data(bk2))
    assert jnp.array_equal(jax.random.key_data(ek1), jax.random.key_data(ek2))
    assert jnp.array_equal(jax.random.key_data(fk1), jax.random.key_data(fk2))
    assert not jnp.array_equal(jax.random.key_data(bk1), jax.random.key_data(ek1)), (
        "build_keys and env_keys must be distinct sub-streams"
    )


# ---------------------------------------------------------------------------
# Unit: stacked pytree shape contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stacked_pytree_has_n_seeds_leading_axis():
    """Every array leaf of the stacked agent + carry carries a leading seed axis."""
    # Given: a trainer with n_seeds=4
    env = GymnaxEnv("CartPole-v1")
    n_seeds = 4
    trainer = MultiSeedTrainer(
        _make_pg_factory(),
        env,
        num_steps=256,
        n_seeds=n_seeds,
        checkpoint_steps=256,
        batch_size=32,
    )

    # When: we build the stacked initial state
    agent_stack, carry = trainer.make_initial_state(jax.random.key(0))

    # Then: every array leaf has leading axis == n_seeds
    agent_arrays, _agent_static = eqx.partition(agent_stack, eqx.is_array)
    for leaf in jax.tree.leaves(agent_arrays):
        assert leaf.shape[0] == n_seeds, f"Agent leaf has wrong leading axis: {leaf.shape}, want leading {n_seeds}"

    for leaf in jax.tree.leaves(carry.agent_state):
        assert leaf.shape[0] == n_seeds, f"agent_state leaf shape {leaf.shape}"
    for leaf in jax.tree.leaves(carry.env_state):
        assert leaf.shape[0] == n_seeds, f"env_state leaf shape {leaf.shape}"
    for leaf in jax.tree.leaves(carry.buffer):
        assert leaf.shape[0] == n_seeds, f"buffer leaf shape {leaf.shape}"
    assert carry.step_count.shape == (n_seeds,)
    assert jax.random.key_data(carry.key).shape[0] == n_seeds


@pytest.mark.unit
def test_static_fields_broadcast_not_stacked():
    """Static (hyperparameter) fields on the agent module survive vmap unchanged."""
    # Given: a trainer with n_seeds=3
    env = GymnaxEnv("CartPole-v1")
    trainer = MultiSeedTrainer(
        _make_ppo_factory(),
        env,
        num_steps=256,
        n_seeds=3,
        checkpoint_steps=256,
        batch_size=32,
    )

    # When: we build the stacked agent
    agent_stack, _carry = trainer.make_initial_state(jax.random.key(0))

    # Then: hyperparameter static fields are plain Python floats, not arrays
    # (PPO declares its hyperparameters as static eqx fields).
    assert isinstance(agent_stack.gamma, float)
    assert isinstance(agent_stack.eps_clip, float)
    assert isinstance(agent_stack.num_epochs, int)


# ---------------------------------------------------------------------------
# Unit: fit return contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fit_returns_seed_indexed_state_map():
    """``fit`` returns a ``{0, 1, ..., n_seeds - 1}``-keyed map of TrainStates."""
    # Given: a trainer with n_seeds=3 on the smallest viable PG config
    env = GymnaxEnv("CartPole-v1")
    n_seeds = 3
    trainer = MultiSeedTrainer(
        _make_pg_factory(),
        env,
        num_steps=256,
        n_seeds=n_seeds,
        checkpoint_steps=256,
        batch_size=32,
    )

    # When: we fit
    states = trainer.fit(jax.random.key(42))

    # Then: keys are 0..n_seeds-1; values look like TrainStates (have ``params``)
    assert set(states.keys()) == set(range(n_seeds)), f"Got keys: {list(states.keys())}"
    for i in range(n_seeds):
        assert hasattr(states[i], "params"), f"Seed {i} state missing 'params'"


# ---------------------------------------------------------------------------
# Integration: two seeds train and differ
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_two_seed_ppo_produces_distinct_finite_states():
    """Two seeds of PPO on CartPole train without NaNs and end at distinct params."""
    # Given: a 2-seed PPO trainer on CartPole
    env = GymnaxEnv("CartPole-v1")
    trainer = MultiSeedTrainer(
        _make_ppo_factory(),
        env,
        num_steps=PPO.collect_size * 2,  # at least 2 learn steps per seed
        n_seeds=2,
        checkpoint_steps=PPO.collect_size,
        batch_size=32,
    )

    # When: we fit
    states = trainer.fit(jax.random.key(7))

    # Then: both states are finite and distinct
    for i in range(2):
        for leaf in jax.tree.leaves(states[i].params):
            assert jnp.all(jnp.isfinite(leaf)), f"Seed {i} produced non-finite params"

    leaves_0 = jax.tree.leaves(states[0].params)
    leaves_1 = jax.tree.leaves(states[1].params)
    any_differ = any(not jnp.allclose(a, b) for a, b in zip(leaves_0, leaves_1, strict=True))
    assert any_differ, "Two seeds produced identical params — vmap axis collapsed?"


@pytest.mark.integration
def test_multi_seed_matches_sequential_trainer_per_seed():
    """``MultiSeedTrainer.fit`` matches ``Trainer.fit`` per seed for identical sub-keys.

    Uses the same per-seed key splitting (``_split_master_key``) for the
    reference Trainer runs so the comparison is apples-to-apples.
    """
    # Given: identical agent factory, env, and shared master key
    env = GymnaxEnv("CartPole-v1")
    factory = _make_pg_factory()
    n_seeds = 2
    num_steps = 512
    batch_size = 32
    master_key = jax.random.key(123)

    mst = MultiSeedTrainer(
        factory,
        env,
        num_steps=num_steps,
        n_seeds=n_seeds,
        checkpoint_steps=num_steps,
        batch_size=batch_size,
    )

    # When: run multi-seed, then run sequential Trainer with the same per-seed sub-keys
    multi_states = mst.fit(master_key)

    build_keys, env_keys, fit_keys = mst._split_master_key(master_key)
    sequential_states = []
    for s in range(n_seeds):
        seed_agent = factory(build_keys[s])
        single = Trainer(
            seed_agent,
            env,
            num_steps=num_steps,
            checkpoint_steps=num_steps,
            batch_size=batch_size,
        )
        # Manually build carry using the same env_key and fit_key so the
        # randomness streams match the vmapped path exactly. The
        # sequential Trainer doesn't expose env/agent keys separately, so
        # construct the carry directly.
        from rltrain.buffer import make_buffer
        from rltrain.trainer._carry import TrainCarry

        env_state = env.reset(env_keys[s])
        carry = TrainCarry(
            agent_state=seed_agent.init(fit_keys[s]),
            env_state=env_state,
            buffer=make_buffer(single.buffer_capacity, env.obs_shape, single.action_shape),
            step_count=jnp.array(0, dtype=jnp.int32),
            key=fit_keys[s],
        )
        sequential_states.append(single.fit(fit_keys[s], carry=carry))

    # Then: per-seed final params match (up to numerical tolerance)
    for s in range(n_seeds):
        multi_leaves = jax.tree.leaves(multi_states[s].params)
        seq_leaves = jax.tree.leaves(sequential_states[s].params)
        for ml, sl in zip(multi_leaves, seq_leaves, strict=True):
            assert ml.shape == sl.shape
            assert jnp.allclose(ml, sl, atol=1e-4, rtol=1e-4), (
                f"Seed {s}: multi-seed vs sequential params differ beyond tol "
                f"(max abs diff = {float(jnp.max(jnp.abs(ml - sl))):.6e})"
            )


@pytest.mark.integration
def test_per_seed_run_dirs_written(tmp_path: Path):
    """Each seed's callbacks write artefacts under ``run_dir/seed_{i}/``."""
    # Given: a 3-seed trainer with a CSVLoggerCallback
    from rltrain.callbacks.csv_logger import CSVLoggerCallback

    env = GymnaxEnv("CartPole-v1")
    n_seeds = 3
    trainer = MultiSeedTrainer(
        _make_pg_factory(),
        env,
        num_steps=512,
        n_seeds=n_seeds,
        checkpoint_steps=256,
        batch_size=32,
        callbacks=[CSVLoggerCallback()],
        run_dir=tmp_path,
    )

    # When: we fit
    trainer.fit(jax.random.key(0))

    # Then: each seed directory exists and contains metrics.csv
    for i in range(n_seeds):
        seed_dir = tmp_path / f"seed_{i}"
        assert seed_dir.is_dir(), f"Missing seed_{i} directory"
        assert (seed_dir / "metrics.csv").is_file(), f"Missing metrics.csv in seed_{i}"


# ---------------------------------------------------------------------------
# Benchmark: per-seed wall clock improves with vmap
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_vmap_seed_axis_improves_per_seed_wall_clock():
    """Per-seed steady-state time at n_seeds=4 is lower than at n_seeds=1.

    Asserts that vmap amortises FLOPs across the seed axis. The threshold
    is conservative (n=4 per-seed < n=1 per-seed) so the test isn't flaky
    on slow CI hardware; the bench script in ``tmp/`` shows ~4-8x.
    """
    env = GymnaxEnv("CartPole-v1")
    factory = _make_ppo_factory()
    num_steps = 2048

    def time_fit(n_seeds: int) -> float:
        trainer = MultiSeedTrainer(
            factory,
            env,
            num_steps=num_steps,
            n_seeds=n_seeds,
            checkpoint_steps=num_steps,
            batch_size=32,
        )
        # Warmup (untimed) to absorb compile cost
        states = trainer.fit(jax.random.key(0))
        jax.block_until_ready(jax.tree.leaves(states[0].params)[0])

        # Steady-state timing — three repeats
        ts = []
        for _ in range(3):
            t0 = time.time()
            states = trainer.fit(jax.random.key(0))
            jax.block_until_ready(jax.tree.leaves(states[0].params)[0])
            ts.append(time.time() - t0)
        return min(ts) / n_seeds  # per-seed wall clock

    per_seed_n1 = time_fit(1)
    per_seed_n4 = time_fit(4)

    print(
        f"\n[bench] per-seed n=1: {per_seed_n1 * 1e3:.1f} ms, "
        f"per-seed n=4: {per_seed_n4 * 1e3:.1f} ms, "
        f"speedup: {per_seed_n1 / per_seed_n4:.2f}x"
    )
    assert per_seed_n4 < per_seed_n1, (
        f"vmap-over-seeds should reduce per-seed wall-clock: n=1 {per_seed_n1:.4f}s, n=4 {per_seed_n4:.4f}s"
    )
