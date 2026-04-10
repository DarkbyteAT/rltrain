r"""Action heads — output parameterisations mapping features to distributions.

Each head is an ``eqx.Module`` that maps network features to a
``distreqx.Distribution``. Agents are agnostic to the action space;
swapping the head switches between discrete and continuous control.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from distreqx.bijectors import Tanh
from distreqx.distributions import Categorical, Normal, Transformed
from jaxtyping import Array, Float, PRNGKeyArray


class DiscreteHead(eqx.Module):
    """Maps features to a Categorical distribution over discrete actions."""

    linear: eqx.nn.Linear

    def __init__(self, feature_dim: int, action_dim: int, *, key: PRNGKeyArray):
        self.linear = eqx.nn.Linear(feature_dim, action_dim, key=key)

    def __call__(self, features: Float[Array, " d"]) -> Categorical:
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
        k1, k2 = jax.random.split(key)
        self.mu_linear = eqx.nn.Linear(feature_dim, action_dim, key=k1)
        self.log_sigma_linear = eqx.nn.Linear(feature_dim, action_dim, key=k2)

    def __call__(self, features: Float[Array, " d"]) -> Normal:
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
        self.gaussian = GaussianHead(feature_dim, action_dim, key=key)

    def __call__(self, features: Float[Array, " d"]) -> Transformed:
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
        k1, k2 = jax.random.split(key)
        self.alpha_linear = eqx.nn.Linear(feature_dim, action_dim, key=k1)
        self.beta_linear = eqx.nn.Linear(feature_dim, action_dim, key=k2)

    def __call__(self, features: Float[Array, " d"]):
        # Defer Gamma import — not all distreqx builds include it
        from distreqx.distributions import Gamma

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
        k1, k2 = jax.random.split(key)
        self.alpha_linear = eqx.nn.Linear(feature_dim, action_dim, key=k1)
        self.beta_linear = eqx.nn.Linear(feature_dim, action_dim, key=k2)

    def __call__(self, features: Float[Array, " d"]):
        from distreqx.distributions import Beta

        alpha = jax.nn.softplus(self.alpha_linear(features)) + 1.0
        beta = jax.nn.softplus(self.beta_linear(features)) + 1.0
        return Beta(alpha=alpha, beta=beta)
