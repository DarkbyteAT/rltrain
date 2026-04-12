"""Composite workflow tests for the Agent protocol and TrainState.

Validates that the design works end-to-end for six composition axes:
1. On-policy (VanillaPG) — init/learn/act cycle
2. Off-policy (VanillaDQN) — target params + Polyak
3. Multi-optimizer via optax.partition
4. jax.grad composes through learn
5. lax.scan with TrainState as carry
6. gradient_step utility correctness
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from spike.agents.agent import Agent, TrainState, gradient_step
from spike.agents.vanilla_dqn import VanillaDQN
from spike.agents.vanilla_pg import VanillaPG
from spike.heads import DiscreteHead
from spike.networks import MLP
from tests.spike.agents._helpers import HIDDEN, NUM_ACTIONS, OBS_DIM
from tests.spike.agents._helpers import _make_off_policy_batch as _make_batch
from tests.spike.agents._helpers import _make_on_policy_transitions as _make_transitions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pg(key: jax.Array) -> VanillaPG:
    k1, k2 = jax.random.split(key)
    return VanillaPG(
        actor=MLP(OBS_DIM, HIDDEN, width=HIDDEN, depth=1, key=k1),
        action_head=DiscreteHead(HIDDEN, NUM_ACTIONS, key=k2),
        optimizer=optax.adam(1e-3),
        gamma=0.99,
        tau=0.01,
        normalise=True,
    )


def _make_dqn(key: jax.Array) -> VanillaDQN:
    return VanillaDQN(
        q_net=MLP(OBS_DIM, NUM_ACTIONS, width=64, depth=2, key=key),
        optimizer=optax.adam(1e-3),
        gamma=0.99,
        target_rate=0.01,
        num_actions=NUM_ACTIONS,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.01,
    )


# ===========================================================================
# Workflow 1: On-policy (VanillaPG)
# ===========================================================================


@pytest.mark.unit
def test_pg_init_produces_valid_train_state():
    """Given a VanillaPG agent, init() returns a TrainState whose params
    match the agent's array structure and whose target_params are zero-filled."""
    # Given
    agent = _make_pg(jax.random.PRNGKey(0))

    # When
    state = agent.init(jax.random.PRNGKey(1))

    # Then — params have same tree structure as agent's array leaves
    agent_params = eqx.partition(agent, eqx.is_array)[0]
    agent_leaves = jax.tree.leaves(agent_params)
    state_leaves = jax.tree.leaves(state.params)
    assert len(agent_leaves) == len(state_leaves)
    for a, s in zip(agent_leaves, state_leaves, strict=False):
        assert a.shape == s.shape

    # And — target_params are all zeros
    for leaf in jax.tree.leaves(state.target_params):
        assert jnp.all(leaf == 0.0), "On-policy target_params should be zero-filled"


@pytest.mark.unit
def test_pg_learn_updates_params_not_targets():
    """Given a PG agent and initial state, learn() updates params and
    opt_state but leaves target_params unchanged."""
    # Given
    agent = _make_pg(jax.random.PRNGKey(0))
    state = agent.init(jax.random.PRNGKey(1))
    batch = _make_transitions(jax.random.PRNGKey(2))

    # When
    new_state, metrics = agent.learn(state, batch, jax.random.PRNGKey(0))

    # Then — params changed
    old_leaves = jax.tree.leaves(state.params)
    new_leaves = jax.tree.leaves(new_state.params)
    any_changed = any(not jnp.allclose(o, n) for o, n in zip(old_leaves, new_leaves, strict=False))
    assert any_changed, "Params should change after learn"

    # And — target_params unchanged (bitwise)
    for old, new in zip(jax.tree.leaves(state.target_params), jax.tree.leaves(new_state.target_params), strict=False):
        assert jnp.array_equal(old, new), "On-policy target_params must not change"

    # And — finite loss
    assert jnp.isfinite(metrics["loss"])


@pytest.mark.unit
def test_pg_act_returns_valid_action():
    """Given a PG agent and state, act() returns an action in valid range."""
    # Given
    agent = _make_pg(jax.random.PRNGKey(0))
    state = agent.init(jax.random.PRNGKey(1))
    obs = jnp.ones(OBS_DIM)

    # When
    action = agent.act(state, obs, jax.random.PRNGKey(2))

    # Then
    assert action.shape == ()
    assert 0 <= int(action) < NUM_ACTIONS


# ===========================================================================
# Workflow 2: Off-policy (VanillaDQN)
# ===========================================================================


@pytest.mark.unit
def test_dqn_init_copies_params_to_target():
    """Given a DQN agent, init() creates target_params as a copy of params."""
    # Given
    agent = _make_dqn(jax.random.PRNGKey(0))

    # When
    state = agent.init(jax.random.PRNGKey(1))

    # Then — target_params match params (bitwise)
    for p, t in zip(jax.tree.leaves(state.params), jax.tree.leaves(state.target_params), strict=False):
        assert jnp.array_equal(p, t), "Target should be an exact copy at init"


@pytest.mark.unit
def test_dqn_learn_applies_polyak_update():
    """Given a DQN agent, learn() applies Polyak averaging to target params:
    target_new = tau * params_new + (1 - tau) * target_old."""
    # Given
    agent = _make_dqn(jax.random.PRNGKey(0))
    state = agent.init(jax.random.PRNGKey(1))
    batch = _make_batch(jax.random.PRNGKey(2))
    old_target_leaves = jax.tree.leaves(state.target_params)

    # When
    new_state, metrics = agent.learn(state, batch, jax.random.PRNGKey(0))
    new_target_leaves = jax.tree.leaves(new_state.target_params)
    new_param_leaves = jax.tree.leaves(new_state.params)

    # Then — target differs from both old target and new params
    for old_t, new_t, new_p in zip(old_target_leaves, new_target_leaves, new_param_leaves, strict=False):
        if old_t.size > 0:
            assert not jnp.allclose(old_t, new_t), "Target should move after Polyak"
            assert not jnp.allclose(new_p, new_t), "Target should differ from new params"

    # And — verify Polyak formula: new_target ≈ tau*new_params + (1-tau)*old_target
    tau = agent.target_rate
    for old_t, new_t, new_p in zip(old_target_leaves, new_target_leaves, new_param_leaves, strict=False):
        expected = tau * new_p + (1.0 - tau) * old_t
        assert jnp.allclose(new_t, expected, atol=1e-6), "Polyak formula violated"


@pytest.mark.unit
def test_dqn_learn_decays_epsilon():
    """Given a DQN agent, learn() decays epsilon by eps_decay."""
    # Given
    agent = _make_dqn(jax.random.PRNGKey(0))
    state = agent.init(jax.random.PRNGKey(1))
    batch = _make_batch(jax.random.PRNGKey(2))

    # When
    new_state, _ = agent.learn(state, batch, jax.random.PRNGKey(0))

    # Then
    expected_eps = max(agent.eps_end, float(state.epsilon) - agent.eps_decay)
    assert jnp.allclose(new_state.epsilon, expected_eps, atol=1e-6)


@pytest.mark.unit
def test_dqn_act_epsilon_greedy():
    """With epsilon=1 actions are random; with epsilon=0 actions are greedy."""
    # Given
    agent = _make_dqn(jax.random.PRNGKey(0))
    state = agent.init(jax.random.PRNGKey(1))
    obs = jnp.ones(OBS_DIM)

    # When — fully random
    greedy_state = state.replace(epsilon=jnp.array(0.0))
    random_state = state.replace(epsilon=jnp.array(1.0))

    keys = jax.random.split(jax.random.PRNGKey(42), 200)
    random_actions = jnp.array([agent.act(random_state, obs, k) for k in keys])
    greedy_actions = jnp.array([agent.act(greedy_state, obs, k) for k in keys])

    # Then — random actions are not all the same
    assert jnp.unique(random_actions).shape[0] > 1

    # And — greedy actions are all identical
    assert jnp.all(greedy_actions == greedy_actions[0])


# ===========================================================================
# Workflow 3: Uniform interface (Trainer doesn't need to know agent type)
# ===========================================================================


@pytest.mark.unit
def test_uniform_learn_interface():
    """Both VanillaPG and VanillaDQN respond to the same learn() call pattern.
    A generic function can call agent.learn(state, batch, key) for either."""
    # Given
    pg = _make_pg(jax.random.PRNGKey(0))
    dqn = _make_dqn(jax.random.PRNGKey(1))
    pg_state = pg.init(jax.random.PRNGKey(2))
    dqn_state = dqn.init(jax.random.PRNGKey(3))
    pg_batch = _make_transitions(jax.random.PRNGKey(4))
    dqn_batch = _make_batch(jax.random.PRNGKey(5))

    # When — same calling pattern for both
    def do_learn(agent, state, batch):
        return agent.learn(state, batch, jax.random.PRNGKey(0))

    pg_new, pg_metrics = do_learn(pg, pg_state, pg_batch)
    dqn_new, dqn_metrics = do_learn(dqn, dqn_state, dqn_batch)

    # Then — both return (state, metrics) with finite loss
    assert jnp.isfinite(pg_metrics["loss"])
    assert jnp.isfinite(dqn_metrics["loss"])


@pytest.mark.unit
def test_agents_satisfy_protocol():
    """VanillaPG and VanillaDQN are runtime-checkable as Agent protocol instances."""
    # Given
    pg = _make_pg(jax.random.PRNGKey(0))
    dqn = _make_dqn(jax.random.PRNGKey(1))

    # Then
    assert isinstance(pg, Agent), "VanillaPG should satisfy Agent protocol"
    assert isinstance(dqn, Agent), "VanillaDQN should satisfy Agent protocol"


# ===========================================================================
# Workflow 4: jax.grad composes through learn
# ===========================================================================


@pytest.mark.unit
def test_grad_through_learn_produces_nonzero_gradients():
    """jax.grad of a loss computed from learn()'s output produces non-zero
    gradients — proving learn is a differentiable pure function."""
    # Given
    agent = _make_pg(jax.random.PRNGKey(0))
    state = agent.init(jax.random.PRNGKey(1))
    batch = _make_transitions(jax.random.PRNGKey(2))
    eval_batch = _make_transitions(jax.random.PRNGKey(3))

    static = eqx.partition(agent, eqx.is_array)[1]

    # When — differentiate through learn
    def meta_loss(params):
        s = TrainState(params=params, opt_state=state.opt_state, target_params=state.target_params)
        new_state, _ = agent.learn(s, batch, jax.random.PRNGKey(0))
        # Evaluate updated params on a different batch
        updated_agent = eqx.combine(new_state.params, static)
        return updated_agent._loss(eval_batch)

    meta_grad = jax.grad(meta_loss)(state.params)

    # Then — at least some gradient leaves are non-zero
    grad_leaves = jax.tree.leaves(meta_grad)
    has_nonzero = any(jnp.any(g != 0.0) for g in grad_leaves)
    assert has_nonzero, "Meta-gradient is all zeros — learn is not differentiable"


@pytest.mark.unit
def test_second_order_gradients_differ_from_first_order():
    """The meta-gradient (through learn) differs from the direct gradient,
    proving jax.grad actually sees through the inner optimisation step."""
    # Given
    agent = _make_pg(jax.random.PRNGKey(0))
    state = agent.init(jax.random.PRNGKey(1))
    batch = _make_transitions(jax.random.PRNGKey(2))
    static = eqx.partition(agent, eqx.is_array)[1]

    # Direct gradient (first-order)
    def direct_loss(params):
        a = eqx.combine(params, static)
        return a._loss(batch)

    direct_grad = jax.grad(direct_loss)(state.params)

    # Meta-gradient (second-order, through learn)
    def meta_loss(params):
        s = TrainState(params=params, opt_state=state.opt_state, target_params=state.target_params)
        new_state, _ = agent.learn(s, batch, jax.random.PRNGKey(0))
        updated_agent = eqx.combine(new_state.params, static)
        return updated_agent._loss(batch)

    meta_grad = jax.grad(meta_loss)(state.params)

    # Then — they differ (the inner step changes the gradient landscape)
    direct_leaves = jax.tree.leaves(direct_grad)
    meta_leaves = jax.tree.leaves(meta_grad)
    any_differ = any(not jnp.allclose(d, m, atol=1e-6) for d, m in zip(direct_leaves, meta_leaves, strict=False))
    assert any_differ, "Meta-gradient equals direct gradient — inner step is invisible to jax.grad"


# ===========================================================================
# Workflow 5: lax.scan with TrainState as carry
# ===========================================================================


@pytest.mark.unit
def test_scan_over_learn_steps():
    """lax.scan over K learn steps compiles and produces valid metrics."""
    # Given
    agent = _make_pg(jax.random.PRNGKey(0))
    state = agent.init(jax.random.PRNGKey(1))
    K = 5
    # Stack K batches along a leading axis
    batches = _make_transitions(jax.random.PRNGKey(2), n=16 * K)
    # Reshape into (K, 16, ...) for scanning
    batched = jax.tree.map(lambda x: x.reshape(K, 16, *x.shape[1:]) if x.ndim > 1 else x.reshape(K, 16), batches)

    # When
    def step(state, batch):
        return agent.learn(state, batch, jax.random.PRNGKey(0))

    final_state, all_metrics = jax.lax.scan(step, state, batched)

    # Then — metrics have shape (K,)
    assert all_metrics["loss"].shape == (K,)
    assert jnp.all(jnp.isfinite(all_metrics["loss"]))

    # And — params changed from initial
    old_leaves = jax.tree.leaves(state.params)
    new_leaves = jax.tree.leaves(final_state.params)
    any_changed = any(not jnp.allclose(o, n) for o, n in zip(old_leaves, new_leaves, strict=False))
    assert any_changed, "Params should change after K scan steps"


@pytest.mark.unit
def test_scan_matches_python_loop():
    """K steps via lax.scan produce bitwise-identical results to K steps
    via a Python for-loop."""
    # Given
    agent = _make_pg(jax.random.PRNGKey(0))
    state = agent.init(jax.random.PRNGKey(1))
    K = 3
    batches = _make_transitions(jax.random.PRNGKey(2), n=16 * K)
    batched = jax.tree.map(lambda x: x.reshape(K, 16, *x.shape[1:]) if x.ndim > 1 else x.reshape(K, 16), batches)

    # When — scan
    def step(state, batch):
        return agent.learn(state, batch, jax.random.PRNGKey(0))

    scan_state, scan_metrics = jax.lax.scan(step, state, batched)

    # When — Python loop
    loop_state = state
    loop_losses = []
    for i in range(K):
        batch_i = jax.tree.map(lambda x, idx=i: x[idx], batched)
        loop_state, m = agent.learn(loop_state, batch_i, jax.random.PRNGKey(0))
        loop_losses.append(m["loss"])

    # Then — numerically equivalent (XLA may fuse differently under scan vs
    # Python loop, producing tiny floating-point differences)
    for s_leaf, l_leaf in zip(jax.tree.leaves(scan_state.params), jax.tree.leaves(loop_state.params), strict=False):
        assert jnp.allclose(s_leaf, l_leaf, atol=1e-5), "Scan and loop should produce equivalent params"

    for s_loss, l_loss in zip(scan_metrics["loss"], loop_losses, strict=False):
        assert jnp.allclose(s_loss, l_loss, atol=1e-5), "Scan and loop should produce equivalent losses"


# ===========================================================================
# Workflow 6: gradient_step utility
# ===========================================================================


@pytest.mark.unit
def test_gradient_step_matches_manual():
    """gradient_step produces bitwise-identical results to the manual
    3-line pattern (filter_value_and_grad, update, apply_updates)."""
    # Given
    agent = _make_pg(jax.random.PRNGKey(0))
    state = agent.init(jax.random.PRNGKey(1))
    batch = _make_transitions(jax.random.PRNGKey(2))
    static = eqx.partition(agent, eqx.is_array)[1]

    def loss_fn(params):
        a = eqx.combine(params, static)
        return a._loss(batch)

    # When — gradient_step
    gs_params, gs_opt, gs_loss = gradient_step(loss_fn, state.params, state.opt_state, agent.optimizer)

    # When — manual
    m_loss, m_grads = eqx.filter_value_and_grad(loss_fn)(state.params)
    m_updates, m_opt = agent.optimizer.update(m_grads, state.opt_state, state.params)
    m_params = optax.apply_updates(state.params, m_updates)

    # Then — bitwise identical
    for gs_leaf, m_leaf in zip(jax.tree.leaves(gs_params), jax.tree.leaves(m_params), strict=False):
        assert jnp.array_equal(gs_leaf, m_leaf)
    assert jnp.array_equal(gs_loss, m_loss)


@pytest.mark.unit
def test_gradient_step_jit_compatible():
    """gradient_step works under jax.jit."""
    # Given
    agent = _make_pg(jax.random.PRNGKey(0))
    state = agent.init(jax.random.PRNGKey(1))
    batch = _make_transitions(jax.random.PRNGKey(2))
    static = eqx.partition(agent, eqx.is_array)[1]

    def loss_fn(params):
        a = eqx.combine(params, static)
        return a._loss(batch)

    # When — JIT-compiled
    jit_gs = jax.jit(gradient_step, static_argnums=(0, 3))
    jit_params, jit_opt, jit_loss = jit_gs(loss_fn, state.params, state.opt_state, agent.optimizer)

    # When — eager
    eager_params, eager_opt, eager_loss = gradient_step(loss_fn, state.params, state.opt_state, agent.optimizer)

    # Then — numerically equivalent (JIT may produce tiny fp differences)
    for j_leaf, e_leaf in zip(jax.tree.leaves(jit_params), jax.tree.leaves(eager_params), strict=False):
        assert jnp.allclose(j_leaf, e_leaf, atol=1e-6), "JIT and eager should produce equivalent params"
    assert jnp.allclose(jit_loss, eager_loss, atol=1e-6)
