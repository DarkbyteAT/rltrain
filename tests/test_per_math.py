"""Math-correctness tests for PER plumbing.

The smoke and convergence tests prove the framework runs and agents
learn. These four close the "is the math actually right?" gap:

1. **PER-uniform equivalence** — every PER-capable agent's
   ``_loss_weighted`` (or weighted critic loss for SAC) must reduce
   numerically to the uniform path when ``is_weights = 1``.
2. **C51 KL non-negativity** — distributional td_errors are KL
   divergences; non-negative by construction iff computed correctly.
3. **Priorities skew sampling** — a slot with priority 1000x higher
   than the others must dominate the sampled indices.

The terminator-freeze test lives in ``tests/agents/test_ppo_terminators.py``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from rltrain.agents.distributional_dqn import DistributionalDQN
from rltrain.agents.double_dqn import DoubleDQN
from rltrain.agents.sac import SAC
from rltrain.agents.vanilla_dqn import VanillaDQN
from rltrain.buffer import buffer_add, buffer_sample, make_buffer
from rltrain.heads import CategoricalAtomHead, SquashedGaussianHead
from rltrain.networks import MLP
from rltrain.transitions import make_transition


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


OBS_DIM = 4
NUM_ACTIONS = 2
ACTION_DIM = 2
BATCH = 16
ATOL = 1e-5  # generous: agents do reductions in float32


def _make_discrete_batch(key, n=BATCH):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return make_transition(
        obs=jax.random.normal(k1, (n, OBS_DIM)),
        action=jax.random.randint(k2, (n,), 0, NUM_ACTIONS),
        reward=jax.random.normal(k3, (n,)),
        next_obs=jax.random.normal(k4, (n, OBS_DIM)),
        done=jnp.zeros(n, dtype=jnp.bool_),
        is_weights=jnp.ones(n),
        indices=jnp.arange(n, dtype=jnp.int32),
    )


def _make_continuous_batch(key, n=BATCH):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return make_transition(
        obs=jax.random.normal(k1, (n, OBS_DIM)),
        action=jax.random.normal(k2, (n, ACTION_DIM)),
        reward=jax.random.normal(k3, (n,)),
        next_obs=jax.random.normal(k4, (n, OBS_DIM)),
        done=jnp.zeros(n, dtype=jnp.bool_),
        is_weights=jnp.ones(n),
        indices=jnp.arange(n, dtype=jnp.int32),
    )


def _make_vanilla_dqn(key):
    return VanillaDQN(
        q_net=MLP(OBS_DIM, NUM_ACTIONS, width_size=32, depth=1, key=key),
        optimizer=optax.adam(1e-3),
        gamma=0.99,
        target_rate=0.005,
        num_actions=NUM_ACTIONS,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=1e-4,
    )


def _make_double_dqn(key):
    return DoubleDQN(
        q_net=MLP(OBS_DIM, NUM_ACTIONS, width_size=32, depth=1, key=key),
        optimizer=optax.adam(1e-3),
        gamma=0.99,
        target_rate=0.005,
        num_actions=NUM_ACTIONS,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=1e-4,
    )


def _make_c51(key):
    k1, k2 = jax.random.split(key)
    return DistributionalDQN(
        feature_net=MLP(OBS_DIM, 32, width_size=32, depth=1, key=k1),
        atom_head=CategoricalAtomHead(32, NUM_ACTIONS, num_atoms=11, v_min=-5, v_max=5, key=k2),
        optimizer=optax.adam(1e-3),
        gamma=0.99,
        target_rate=0.005,
        num_actions=NUM_ACTIONS,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=1e-4,
    )


def _make_continuous_sac(key):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return SAC(
        actor=MLP(OBS_DIM, 32, width_size=32, depth=1, key=k1),
        action_head=SquashedGaussianHead(32, ACTION_DIM, key=k2),
        critic_1=MLP(OBS_DIM + ACTION_DIM, 1, width_size=32, depth=1, key=k3),
        critic_2=MLP(OBS_DIM + ACTION_DIM, 1, width_size=32, depth=1, key=k4),
        actor_optimizer=optax.adam(3e-4),
        critic_optimizer=optax.adam(3e-4),
        alpha_optimizer=optax.adam(3e-4),
        gamma=0.99,
        tau=0.005,
    )


# ---------------------------------------------------------------------------
# Test 1: PER-uniform equivalence
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_vanilla_dqn_uniform_equivalence():
    """``_loss_weighted`` with is_weights=1 equals ``_loss`` for VanillaDQN."""
    # Given a VanillaDQN, its state, and a batch with unit IS weights
    agent = _make_vanilla_dqn(jax.random.PRNGKey(0))
    state = agent.init(jax.random.PRNGKey(1))
    batch = _make_discrete_batch(jax.random.PRNGKey(2))
    static = eqx.partition(agent, eqx.is_array)[1]
    model = eqx.combine(state.params, static)

    # When both paths are evaluated
    uniform = model._loss(state.target_params, static, batch)
    weighted, aux = model._loss_weighted(state.target_params, static, batch, batch.is_weights)

    # Then they agree numerically
    assert jnp.allclose(uniform, weighted, atol=ATOL), f"VanillaDQN uniform={uniform} weighted={weighted}"
    # And td_errors are finite + non-negative (absolute values)
    assert jnp.all(jnp.isfinite(aux["td_errors"]))
    assert jnp.all(aux["td_errors"] >= 0.0)


@pytest.mark.unit
def test_double_dqn_uniform_equivalence():
    """``_loss_weighted`` with is_weights=1 equals ``_loss`` for DoubleDQN (double-Q TD)."""
    agent = _make_double_dqn(jax.random.PRNGKey(0))
    state = agent.init(jax.random.PRNGKey(1))
    batch = _make_discrete_batch(jax.random.PRNGKey(2))
    static = eqx.partition(agent, eqx.is_array)[1]
    model = eqx.combine(state.params, static)

    uniform = model._loss(state.target_params, static, batch)
    weighted, aux = model._loss_weighted(state.target_params, static, batch, batch.is_weights)

    assert jnp.allclose(uniform, weighted, atol=ATOL), f"DoubleDQN uniform={uniform} weighted={weighted}"
    assert jnp.all(jnp.isfinite(aux["td_errors"]))
    assert jnp.all(aux["td_errors"] >= 0.0)


@pytest.mark.unit
def test_distributional_dqn_uniform_equivalence():
    """``_loss_weighted`` with is_weights=1 equals ``_loss`` for DistributionalDQN."""
    agent = _make_c51(jax.random.PRNGKey(0))
    state = agent.init(jax.random.PRNGKey(1))
    batch = _make_discrete_batch(jax.random.PRNGKey(2))
    static = eqx.partition(agent, eqx.is_array)[1]
    model = eqx.combine(state.params, static)

    uniform = model._loss(state.target_params, static, batch)
    weighted, aux = model._loss_weighted(state.target_params, static, batch, batch.is_weights)

    assert jnp.allclose(uniform, weighted, atol=ATOL), f"C51 uniform={uniform} weighted={weighted}"
    assert jnp.all(jnp.isfinite(aux["td_errors"]))


@pytest.mark.unit
def test_sac_critic_uniform_equivalence():
    """SAC's IS-weighted critic loss with is_weights=1 equals the plain twin-mean Bellman MSE."""
    # Given a continuous SAC and a batch with unit IS weights
    agent = _make_continuous_sac(jax.random.PRNGKey(0))
    state = agent.init(jax.random.PRNGKey(1))
    batch = _make_continuous_batch(jax.random.PRNGKey(2))
    actor_static, critic_static = agent._statics()
    alpha = jnp.exp(state.log_alpha)
    key = jax.random.PRNGKey(3)

    # When the new IS-weighted critic loss runs with is_weights=1
    weighted_loss, aux = agent._critic_loss(
        state.critic_params,
        critic_static,
        state.target_critic_params,
        state.actor_params,
        actor_static,
        alpha,
        batch,
        key,
    )

    # And the same arithmetic is computed by hand without IS weighting:
    # mean over the batch of 0.5 * (twin-1 SE + twin-2 SE)
    q1, q2 = agent._reconstruct_critics(state.critic_params, critic_static)
    t_q1, t_q2 = agent._reconstruct_critics(state.target_critic_params, critic_static)
    actor, head = eqx.combine(state.actor_params, actor_static)

    q1_val = jax.vmap(lambda o, a: q1(jnp.concatenate([o, a])).squeeze())(batch.obs, batch.action)
    q2_val = jax.vmap(lambda o, a: q2(jnp.concatenate([o, a])).squeeze())(batch.obs, batch.action)
    next_features = jax.vmap(actor)(batch.next_obs)
    next_dists = jax.vmap(head)(next_features)
    next_actions, next_per_dim_lp = next_dists.sample_and_log_prob(key)
    next_log_probs = jnp.sum(next_per_dim_lp, axis=-1)
    t_q1_next = jax.vmap(lambda o, a: t_q1(jnp.concatenate([o, a])).squeeze())(batch.next_obs, next_actions)
    t_q2_next = jax.vmap(lambda o, a: t_q2(jnp.concatenate([o, a])).squeeze())(batch.next_obs, next_actions)
    min_q_next = jnp.minimum(t_q1_next, t_q2_next)
    v_next = min_q_next - alpha * next_log_probs
    done_mask = 1.0 - batch.done.astype(jnp.float32)
    td_target = batch.reward + agent.gamma * done_mask * v_next
    reference = jnp.mean(0.5 * ((q1_val - td_target) ** 2 + (q2_val - td_target) ** 2))

    # Then the IS-weighted result with is_weights=1 matches the uniform reference
    assert jnp.allclose(weighted_loss, reference, atol=ATOL), (
        f"SAC critic weighted={weighted_loss} reference={reference}"
    )
    # td_errors are finite + non-negative
    assert jnp.all(jnp.isfinite(aux["td_errors"]))
    assert jnp.all(aux["td_errors"] >= 0.0)


# ---------------------------------------------------------------------------
# Test 2: C51 KL non-negativity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_c51_td_errors_are_kl_non_negative():
    """C51's td_errors are per-sample KL divergences — must be non-negative."""
    # Given C51, state, a random batch
    agent = _make_c51(jax.random.PRNGKey(0))
    state = agent.init(jax.random.PRNGKey(1))
    batch = _make_discrete_batch(jax.random.PRNGKey(2))
    static = eqx.partition(agent, eqx.is_array)[1]
    model = eqx.combine(state.params, static)

    # When td_errors are computed via the weighted path
    _loss, aux = model._loss_weighted(state.target_params, static, batch, batch.is_weights)
    td_errors = aux["td_errors"]

    # Then they're per-sample (shape (B,)), finite, and KL >= 0 elementwise.
    # A small negative tolerance absorbs float32 rounding around zero.
    assert td_errors.shape == (BATCH,), f"td_errors shape {td_errors.shape}, expected ({BATCH},)"
    assert jnp.all(jnp.isfinite(td_errors))
    assert jnp.all(td_errors >= -1e-6), f"KL must be non-negative; got min={float(jnp.min(td_errors))}"


# ---------------------------------------------------------------------------
# Test 3: Priorities skew sampling
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_priorities_skew_sampling_toward_high_slot():
    """A slot with priority 1000x higher than others dominates the sampled indices."""
    # Given a full buffer of 100 transitions with priority[0]=1000, priority[1:]=1.0
    capacity = 100
    buf = make_buffer(capacity=capacity, obs_shape=(OBS_DIM,), action_shape=())
    for i in range(capacity):
        t = make_transition(
            obs=jnp.ones(OBS_DIM) * i,
            action=jnp.array(0),
            reward=jnp.array(float(i)),
            next_obs=jnp.ones(OBS_DIM) * (i + 1),
            done=jnp.array(False),
        )
        buf = buffer_add(buf, t)
    skewed = buf.priorities.at[0].set(1000.0)
    buf = buf.replace(priorities=skewed)

    # When 100 samples are drawn via prioritised sampling
    _batch, indices, _is_weights = buffer_sample(
        buf, jax.random.PRNGKey(42), batch_size=100, prioritised=True, alpha=1.0
    )

    # Then slot 0 is heavily over-represented. With alpha=1 and priority ratio
    # 1000:1 against 99 unit-priority slots, p(slot 0) ≈ 1000 / (1000 + 99) ≈ 0.91,
    # so an expected count near ~91 out of 100. A conservative threshold catches
    # any sampling-skew bug well above uniform noise.
    count_zero = int(jnp.sum(indices == 0))
    assert count_zero > 30, f"slot 0 sampled only {count_zero}/100 times; expected dominance from priority 1000:1"
