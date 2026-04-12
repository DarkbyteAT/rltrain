r"""Action heads — output parameterisations mapping features to distributions.

Each head is an ``eqx.Module`` that maps network features to a
``distreqx.Distribution``. Agents are agnostic to the action space;
swapping the head switches between discrete and continuous control.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from distreqx.bijectors import Tanh
from distreqx.distributions import Beta, Categorical, Gamma, Normal, Transformed
from jaxtyping import Array, Float, PRNGKeyArray


class DiscreteHead(eqx.Module):
    """Maps features to a Categorical distribution over discrete actions."""

    linear: eqx.nn.Linear

    def __init__(self, feature_dim: int, action_dim: int, *, key: PRNGKeyArray):
        """Initialise with a single linear layer projecting to action logits."""
        self.linear = eqx.nn.Linear(feature_dim, action_dim, key=key)

    def __call__(self, features: Float[Array, " d"]) -> Categorical:
        """Map features to a Categorical distribution."""
        logits = self.linear(features)
        return Categorical(logits=logits)


class GaussianHead(eqx.Module):
    r"""Maps features to an unbounded Normal distribution.

    Two output heads: $\mu$ (unconstrained) and $\log \sigma$ (clipped for
    numerical stability).
    """

    mu_linear: eqx.nn.Linear
    log_sigma_linear: eqx.nn.Linear
    log_sigma_min: float = eqx.field(static=True, default=-20.0)
    log_sigma_max: float = eqx.field(static=True, default=2.0)

    def __init__(self, feature_dim: int, action_dim: int, *, key: PRNGKeyArray):
        """Initialise mean and log-std linear layers."""
        k1, k2 = jax.random.split(key)
        self.mu_linear = eqx.nn.Linear(feature_dim, action_dim, key=k1)
        self.log_sigma_linear = eqx.nn.Linear(feature_dim, action_dim, key=k2)

    def __call__(self, features: Float[Array, " d"]) -> Normal:
        """Map features to a diagonal Normal distribution."""
        mu = self.mu_linear(features)
        log_sigma = jnp.clip(self.log_sigma_linear(features), self.log_sigma_min, self.log_sigma_max)
        return Normal(loc=mu, scale=jnp.exp(log_sigma))


class SquashedGaussianHead(eqx.Module):
    r"""Maps features to a tanh-squashed Normal distribution.

    Used by SAC for bounded continuous actions. The ``Transformed`` distribution
    automatically corrects ``log_prob`` for the Jacobian of the tanh bijector.
    """

    gaussian: GaussianHead

    def __init__(self, feature_dim: int, action_dim: int, *, key: PRNGKeyArray):
        """Initialise the underlying Gaussian head."""
        self.gaussian = GaussianHead(feature_dim, action_dim, key=key)

    def __call__(self, features: Float[Array, " d"]) -> Transformed:
        """Map features to a tanh-squashed Normal distribution."""
        base_dist = self.gaussian(features)
        return Transformed(distribution=base_dist, bijector=Tanh())


class GammaHead(eqx.Module):
    r"""Maps features to a Gamma distribution (non-negative continuous actions).

    Outputs concentration $\alpha$ and rate $\beta$ via softplus to ensure
    positivity.
    """

    alpha_linear: eqx.nn.Linear
    beta_linear: eqx.nn.Linear

    def __init__(self, feature_dim: int, action_dim: int, *, key: PRNGKeyArray):
        """Initialise concentration and rate linear layers."""
        k1, k2 = jax.random.split(key)
        self.alpha_linear = eqx.nn.Linear(feature_dim, action_dim, key=k1)
        self.beta_linear = eqx.nn.Linear(feature_dim, action_dim, key=k2)

    def __call__(self, features: Float[Array, " d"]):
        """Map features to a Gamma distribution with positive parameters."""
        alpha = jax.nn.softplus(self.alpha_linear(features))
        beta = jax.nn.softplus(self.beta_linear(features))
        return Gamma(concentration=alpha, rate=beta)


class BetaHead(eqx.Module):
    r"""Maps features to a Beta distribution (bounded [0, 1] continuous actions).

    Outputs $\alpha, \beta > 1$ via ``softplus(x) + 1`` to ensure the
    distribution is unimodal.
    """

    alpha_linear: eqx.nn.Linear
    beta_linear: eqx.nn.Linear

    def __init__(self, feature_dim: int, action_dim: int, *, key: PRNGKeyArray):
        """Initialise alpha and beta linear layers."""
        k1, k2 = jax.random.split(key)
        self.alpha_linear = eqx.nn.Linear(feature_dim, action_dim, key=k1)
        self.beta_linear = eqx.nn.Linear(feature_dim, action_dim, key=k2)

    def __call__(self, features: Float[Array, " d"]):
        """Map features to a Beta distribution with unimodal parameters."""
        alpha = jax.nn.softplus(self.alpha_linear(features)) + 1.0
        beta = jax.nn.softplus(self.beta_linear(features)) + 1.0
        return Beta(alpha=alpha, beta=beta)


class CategoricalAtomHead(eqx.Module):
    r"""Maps features to a categorical distribution over fixed value atoms (C51).

    Outputs a probability mass function over ``num_atoms`` atoms spanning
    $[V_{\min}, V_{\max}]$.  Used by distributional DQN (C51).

    The atoms are stored as a static field and shared with
    ``spike.math.project_distribution`` and ``spike.math.q_values_from_pmf``.
    """

    linear: eqx.nn.Linear
    atoms: Float[Array, " num_atoms"] = eqx.field(static=True)
    num_atoms: int = eqx.field(static=True)
    num_actions: int = eqx.field(static=True)

    def __init__(
        self,
        feature_dim: int,
        num_actions: int,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        *,
        key: PRNGKeyArray,
    ):
        """Initialise with a linear layer projecting to ``num_actions * num_atoms`` logits."""
        self.linear = eqx.nn.Linear(feature_dim, num_actions * num_atoms, key=key)
        self.atoms = jnp.linspace(v_min, v_max, num_atoms)
        self.num_atoms = num_atoms
        self.num_actions = num_actions

    def __call__(self, features: Float[Array, " d"]) -> Float[Array, "num_actions num_atoms"]:
        """Map features to per-action PMFs via softmax over the atom axis."""
        logits = self.linear(features)
        logits = logits.reshape(self.num_actions, self.num_atoms)
        return jax.nn.softmax(logits, axis=-1)
