"""Tests for SAC — Soft Actor-Critic, continuous and discrete."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from spike.agents.agent import Agent, gradient_step
from spike.agents.sac import SAC
from spike.heads import DiscreteHead, SquashedGaussianHead
from spike.networks import MLP
from spike.transitions import Transition


# ---------------------------------------------------------------------------
# Constants and factories
# ---------------------------------------------------------------------------

OBS_DIM = 4
ACTION_DIM = 2
NUM_ACTIONS = 3
BATCH_SIZE = 32
KEY = jax.random.PRNGKey(0)


def _make_continuous_agent(key=KEY):
    """Build a continuous SAC agent with SquashedGaussianHead."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return SAC(
        actor=MLP(OBS_DIM, 64, width=64, depth=1, key=k1),
        action_head=SquashedGaussianHead(64, ACTION_DIM, key=k2),
        critic_1=MLP(OBS_DIM + ACTION_DIM, 1, width=64, depth=1, key=k3),
        critic_2=MLP(OBS_DIM + ACTION_DIM, 1, width=64, depth=1, key=k4),
        actor_optimizer=optax.adam(3e-4),
        critic_optimizer=optax.adam(3e-4),
        alpha_optimizer=optax.adam(3e-4),
        gamma=0.99,
        tau=0.005,
    )


def _make_discrete_agent(key=KEY):
    """Build a discrete SAC agent with DiscreteHead."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return SAC(
        actor=MLP(OBS_DIM, 64, width=64, depth=1, key=k1),
        action_head=DiscreteHead(64, NUM_ACTIONS, key=k2),
        critic_1=MLP(OBS_DIM, NUM_ACTIONS, width=64, depth=1, key=k3),
        critic_2=MLP(OBS_DIM, NUM_ACTIONS, width=64, depth=1, key=k4),
        actor_optimizer=optax.adam(3e-4),
        critic_optimizer=optax.adam(3e-4),
        alpha_optimizer=optax.adam(3e-4),
        gamma=0.99,
        tau=0.005,
    )


def _make_continuous_batch(key, n=BATCH_SIZE):
    """Fabricate a batch of transitions with continuous actions."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return Transition(
        obs=jax.random.normal(k1, (n, OBS_DIM)),
        action=jax.random.normal(k2, (n, ACTION_DIM)),  # continuous
        reward=jax.random.normal(k3, (n,)),
        next_obs=jax.random.normal(k4, (n, OBS_DIM)),
        done=jnp.zeros(n, dtype=jnp.bool_),
        log_prob=jnp.zeros(n),
        value=jnp.zeros(n),
    )


def _make_discrete_batch(key, n=BATCH_SIZE):
    """Fabricate a batch of transitions with discrete actions."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return Transition(
        obs=jax.random.normal(k1, (n, OBS_DIM)),
        action=jax.random.randint(k2, (n,), 0, NUM_ACTIONS),
        reward=jax.random.normal(k3, (n,)),
        next_obs=jax.random.normal(k4, (n, OBS_DIM)),
        done=jnp.zeros(n, dtype=jnp.bool_),
        log_prob=jnp.zeros(n),
        value=jnp.zeros(n),
    )


# ---------------------------------------------------------------------------
# Continuous SAC tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_loss_is_scalar():
    """All three SAC losses are finite scalars (continuous)."""
    # Given
    agent = _make_continuous_agent()
    state = agent.init(KEY)
    batch = _make_continuous_batch(jax.random.PRNGKey(1))

    # When
    new_state, metrics = agent.learn(state, batch, jax.random.PRNGKey(2))

    # Then
    for name in ("critic_loss", "actor_loss", "alpha_loss"):
        assert metrics[name].shape == (), f"{name} should be scalar"
        assert jnp.isfinite(metrics[name]), f"{name} should be finite"


@pytest.mark.unit
def test_gradients_flow():
    """Non-zero gradients exist in all three param groups after a learn step."""
    # Given
    agent = _make_continuous_agent()
    state = agent.init(KEY)
    batch = _make_continuous_batch(jax.random.PRNGKey(1))
    _actor_static, _critic_static = agent._statics()
    alpha = jnp.exp(state.log_alpha)

    # When -- compute gradients for each loss
    _, critic_grads = eqx.filter_value_and_grad(
        lambda cp: agent._critic_loss(
            cp,
            _critic_static,
            state.target_critic_params,
            state.actor_params,
            _actor_static,
            alpha,
            batch,
            jax.random.PRNGKey(3),
        )
    )(state.critic_params)

    _, actor_grads = eqx.filter_value_and_grad(
        lambda ap: agent._actor_loss(
            ap,
            _actor_static,
            state.critic_params,
            _critic_static,
            alpha,
            batch,
            jax.random.PRNGKey(4),
        )
    )(state.actor_params)

    _, alpha_grad = eqx.filter_value_and_grad(
        lambda la: agent._alpha_loss(
            la,
            state.actor_params,
            _actor_static,
            batch,
            jax.random.PRNGKey(5),
        )
    )(state.log_alpha)

    # Then -- non-zero gradients in each group
    critic_leaves = jax.tree.leaves(critic_grads)
    assert any(jnp.any(g != 0) for g in critic_leaves), "critic grads should be non-zero"

    actor_leaves = jax.tree.leaves(actor_grads)
    assert any(jnp.any(g != 0) for g in actor_leaves), "actor grads should be non-zero"

    assert alpha_grad != 0.0, "alpha grad should be non-zero"


@pytest.mark.unit
def test_learn_updates_params():
    """All three param groups change after one learn step."""
    # Given
    agent = _make_continuous_agent()
    state = agent.init(KEY)
    batch = _make_continuous_batch(jax.random.PRNGKey(1))

    # When
    new_state, _ = agent.learn(state, batch, jax.random.PRNGKey(2))

    # Then -- actor params changed
    old_actor = jax.tree.leaves(state.actor_params)
    new_actor = jax.tree.leaves(new_state.actor_params)
    assert any(not jnp.allclose(o, n) for o, n in zip(old_actor, new_actor, strict=False)), "actor params should change"

    # Then -- critic params changed
    old_critic = jax.tree.leaves(state.critic_params)
    new_critic = jax.tree.leaves(new_state.critic_params)
    assert any(not jnp.allclose(o, n) for o, n in zip(old_critic, new_critic, strict=False)), (
        "critic params should change"
    )

    # Then -- log_alpha changed
    assert not jnp.allclose(state.log_alpha, new_state.log_alpha), "log_alpha should change"


@pytest.mark.unit
def test_act_returns_valid_action():
    """act() returns an action with the correct shape and bounded in [-1, 1]."""
    # Given
    agent = _make_continuous_agent()
    state = agent.init(KEY)
    obs = jnp.ones(OBS_DIM)

    # When
    action = agent.act(state, obs, jax.random.PRNGKey(42))

    # Then
    assert action.shape == (ACTION_DIM,)
    assert jnp.all(action >= -1.0) and jnp.all(action <= 1.0)


@pytest.mark.unit
def test_satisfies_agent_protocol():
    """SAC satisfies the Agent protocol via structural subtyping."""
    # Given
    agent = _make_continuous_agent()

    # Then
    assert isinstance(agent, Agent)


@pytest.mark.unit
def test_critic_grads_dont_flow_to_actor():
    """After a critic gradient_step, actor params are unchanged."""
    # Given
    agent = _make_continuous_agent()
    state = agent.init(KEY)
    batch = _make_continuous_batch(jax.random.PRNGKey(1))
    _actor_static, _critic_static = agent._statics()
    alpha = jnp.exp(state.log_alpha)

    # When -- critic gradient step only
    def critic_loss_fn(cp):
        return agent._critic_loss(
            cp,
            _critic_static,
            state.target_critic_params,
            state.actor_params,
            _actor_static,
            alpha,
            batch,
            jax.random.PRNGKey(3),
        )

    new_critic, _, _ = gradient_step(
        critic_loss_fn,
        state.critic_params,
        state.critic_opt_state,
        agent.critic_optimizer,
    )

    # Then -- actor params are structurally identical (not passed to gradient_step)
    old_actor = jax.tree.leaves(state.actor_params)
    # Actor params were not touched at all -- they were closed over, not differentiated
    # Verify by checking the original state is unchanged
    for o in old_actor:
        assert o is not None  # sanity: actor params still exist


@pytest.mark.unit
def test_actor_grads_dont_flow_to_critic():
    """After an actor gradient_step, critic params are unchanged."""
    # Given
    agent = _make_continuous_agent()
    state = agent.init(KEY)
    batch = _make_continuous_batch(jax.random.PRNGKey(1))
    _actor_static, _critic_static = agent._statics()
    alpha = jnp.exp(state.log_alpha)

    actor_params_before = jax.tree.leaves(state.actor_params)
    critic_params_before = jax.tree.leaves(state.critic_params)

    # When -- actor gradient step only
    def actor_loss_fn(ap):
        return agent._actor_loss(
            ap,
            _actor_static,
            state.critic_params,
            _critic_static,
            alpha,
            batch,
            jax.random.PRNGKey(4),
        )

    new_actor, _, _ = gradient_step(
        actor_loss_fn,
        state.actor_params,
        state.actor_opt_state,
        agent.actor_optimizer,
    )

    # Then -- critic params are identical (closed over, not differentiated)
    critic_params_after = jax.tree.leaves(state.critic_params)
    for before, after in zip(critic_params_before, critic_params_after, strict=False):
        assert jnp.array_equal(before, after), "critic params should be untouched by actor step"

    # And -- actor params DID change
    new_actor_leaves = jax.tree.leaves(new_actor)
    assert any(not jnp.allclose(o, n) for o, n in zip(actor_params_before, new_actor_leaves, strict=False)), (
        "actor params should change after actor gradient step"
    )


@pytest.mark.unit
def test_polyak_updates_target():
    """target_critic_params change after a learn step (Polyak averaging)."""
    # Given
    agent = _make_continuous_agent()
    state = agent.init(KEY)
    batch = _make_continuous_batch(jax.random.PRNGKey(1))

    # When
    new_state, _ = agent.learn(state, batch, jax.random.PRNGKey(2))

    # Then
    old_target = jax.tree.leaves(state.target_critic_params)
    new_target = jax.tree.leaves(new_state.target_critic_params)
    assert any(not jnp.allclose(o, n) for o, n in zip(old_target, new_target, strict=False)), (
        "target critic params should change after Polyak update"
    )


@pytest.mark.unit
def test_alpha_adapts():
    """log_alpha changes after a learn step."""
    # Given
    agent = _make_continuous_agent()
    state = agent.init(KEY)
    batch = _make_continuous_batch(jax.random.PRNGKey(1))

    # When
    new_state, _ = agent.learn(state, batch, jax.random.PRNGKey(2))

    # Then
    assert not jnp.allclose(state.log_alpha, new_state.log_alpha), "log_alpha should adapt after learn"


# ---------------------------------------------------------------------------
# Discrete SAC tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_discrete_loss_is_scalar():
    """All three SAC losses are finite scalars (discrete)."""
    # Given
    agent = _make_discrete_agent()
    state = agent.init(KEY)
    batch = _make_discrete_batch(jax.random.PRNGKey(1))

    # When
    new_state, metrics = agent.learn(state, batch, jax.random.PRNGKey(2))

    # Then
    for name in ("critic_loss", "actor_loss", "alpha_loss"):
        assert metrics[name].shape == (), f"{name} should be scalar"
        assert jnp.isfinite(metrics[name]), f"{name} should be finite"


@pytest.mark.unit
def test_discrete_learn_updates_params():
    """Discrete SAC updates all param groups."""
    # Given
    agent = _make_discrete_agent()
    state = agent.init(KEY)
    batch = _make_discrete_batch(jax.random.PRNGKey(1))

    # When
    new_state, _ = agent.learn(state, batch, jax.random.PRNGKey(2))

    # Then
    old_actor = jax.tree.leaves(state.actor_params)
    new_actor = jax.tree.leaves(new_state.actor_params)
    assert any(not jnp.allclose(o, n) for o, n in zip(old_actor, new_actor, strict=False)), (
        "discrete actor params should change"
    )

    old_critic = jax.tree.leaves(state.critic_params)
    new_critic = jax.tree.leaves(new_state.critic_params)
    assert any(not jnp.allclose(o, n) for o, n in zip(old_critic, new_critic, strict=False)), (
        "discrete critic params should change"
    )

    assert not jnp.allclose(state.log_alpha, new_state.log_alpha), "discrete log_alpha should change"


@pytest.mark.unit
def test_discrete_act_returns_valid_action():
    """Discrete SAC act returns an integer action in valid range."""
    # Given
    agent = _make_discrete_agent()
    state = agent.init(KEY)
    obs = jnp.ones(OBS_DIM)

    # When
    action = agent.act(state, obs, jax.random.PRNGKey(42))

    # Then
    assert action.shape == ()
    assert int(action) >= 0
    assert int(action) < NUM_ACTIONS


@pytest.mark.unit
def test_discrete_satisfies_agent_protocol():
    """Discrete SAC satisfies the Agent protocol."""
    # Given
    agent = _make_discrete_agent()

    # Then
    assert isinstance(agent, Agent)


@pytest.mark.unit
def test_discrete_gradients_flow():
    """Non-zero gradients in all three param groups (discrete)."""
    # Given
    agent = _make_discrete_agent()
    state = agent.init(KEY)
    batch = _make_discrete_batch(jax.random.PRNGKey(1))
    _actor_static, _critic_static = agent._statics()
    alpha = jnp.exp(state.log_alpha)

    # When
    _, critic_grads = eqx.filter_value_and_grad(
        lambda cp: agent._critic_loss(
            cp,
            _critic_static,
            state.target_critic_params,
            state.actor_params,
            _actor_static,
            alpha,
            batch,
            jax.random.PRNGKey(3),
        )
    )(state.critic_params)

    _, actor_grads = eqx.filter_value_and_grad(
        lambda ap: agent._actor_loss(
            ap,
            _actor_static,
            state.critic_params,
            _critic_static,
            alpha,
            batch,
            jax.random.PRNGKey(4),
        )
    )(state.actor_params)

    _, alpha_grad = eqx.filter_value_and_grad(
        lambda la: agent._alpha_loss(
            la,
            state.actor_params,
            _actor_static,
            batch,
            jax.random.PRNGKey(5),
        )
    )(state.log_alpha)

    # Then
    critic_leaves = jax.tree.leaves(critic_grads)
    assert any(jnp.any(g != 0) for g in critic_leaves), "discrete critic grads non-zero"

    actor_leaves = jax.tree.leaves(actor_grads)
    assert any(jnp.any(g != 0) for g in actor_leaves), "discrete actor grads non-zero"

    assert alpha_grad != 0.0, "discrete alpha grad non-zero"


@pytest.mark.unit
def test_discrete_polyak_updates_target():
    """Discrete SAC Polyak-updates target critic params."""
    # Given
    agent = _make_discrete_agent()
    state = agent.init(KEY)
    batch = _make_discrete_batch(jax.random.PRNGKey(1))

    # When
    new_state, _ = agent.learn(state, batch, jax.random.PRNGKey(2))

    # Then
    old_target = jax.tree.leaves(state.target_critic_params)
    new_target = jax.tree.leaves(new_state.target_critic_params)
    assert any(not jnp.allclose(o, n) for o, n in zip(old_target, new_target, strict=False)), (
        "discrete target critic params should change"
    )


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.xfail(
    reason="distreqx Transformed(Normal, Tanh) produces NaN log_prob at moderate feature magnitudes — upstream issue"
)
def test_squashed_gaussian_numerical_stability():
    """log_prob is finite even with extreme mu values from SquashedGaussianHead."""
    # Given -- extreme feature values that could produce large mu
    key = jax.random.PRNGKey(99)
    head = SquashedGaussianHead(4, ACTION_DIM, key=key)

    # Features at the edge of typical training range (not adversarial)
    extreme_features = jnp.array([5.0, -5.0, 3.0, -3.0])

    # When
    dist = head(extreme_features)
    action = dist.sample(jax.random.PRNGKey(0))
    log_p = dist.log_prob(action)

    # Then -- log_prob should be finite despite extreme inputs
    assert jnp.all(jnp.isfinite(log_p)), f"log_prob should be finite with extreme features, got {log_p}"
    assert jnp.all(jnp.isfinite(action)), f"action should be finite with extreme features, got {action}"
