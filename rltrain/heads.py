r"""Action heads — output parameterisations mapping features to distributions.

Each head is an ``eqx.Module`` that maps network features to a
``distreqx.Distribution``. Agents are agnostic to the action space;
swapping the head switches between discrete and continuous control.

Divergence from PyTorch rltrain: ``GaussianHead`` uses two separate Linear
layers (one for ``mu``, one for ``log_sigma``) and clips ``log_sigma`` to
``[-20, 2]``. PyTorch rltrain interleaved a single linear output and did
not clip. The clip prevents ``exp(log_sigma)`` from over/underflowing at
the start of training; the cost is the policy can never collapse to a
truly deterministic action. This is the SAC/PPO-standard trade-off and
is the preferred behaviour for continuous-control benchmarks.
"""

from typing import Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
from distreqx.distributions import AbstractDistribution, Beta, Categorical, Gamma, Normal
from jaxtyping import Array, Float, PRNGKeyArray


@runtime_checkable
class Head(Protocol):
    r"""Structural protocol for action heads.

    An action head is any callable mapping a feature vector to a
    ``distreqx`` distribution over actions, and exposing the action
    dimensionality via an ``action_dim`` property. Concrete
    implementations in this module (``DiscreteHead``, ``GaussianHead``,
    ``SquashedGaussianHead``, ``GammaHead``, ``BetaHead``) all satisfy
    this contract; users may supply their own. The
    ``OnPolicyAgent.action_head`` and ``SAC.action_head`` slots accept
    any ``eqx.Module`` matching this shape — ``runtime_checkable`` lets
    ``isinstance`` checks succeed for duck-typed heads.

    Why ``action_dim`` is part of the protocol: SAC reads it at
    construction time to derive a default ``target_entropy``. Without it,
    SAC would have to reach into private fields like ``linear.out_features``
    or ``gaussian.mu_linear.out_features``, coupling itself to each head's
    internal structure and breaking any custom head that doesn't replicate
    those names.

    ``CategoricalAtomHead`` deliberately does NOT conform: it is a *value*
    head returning a PMF array used by distributional DQN, not an action
    distribution.
    """

    @property
    def action_dim(self) -> int:
        """Number of action components produced by this head."""
        ...

    def __call__(self, features: Float[Array, " d"]) -> AbstractDistribution:
        """Map a feature vector to a distribution over actions."""
        ...


class DiscreteHead(eqx.Module):
    """Maps features to a Categorical distribution over discrete actions."""

    linear: eqx.nn.Linear

    def __init__(self, feature_dim: int, action_dim: int, *, key: PRNGKeyArray):
        """Initialise with a single linear layer projecting to action logits."""
        self.linear = eqx.nn.Linear(feature_dim, action_dim, key=key)

    @property
    def action_dim(self) -> int:
        """Number of discrete actions (categorical support size)."""
        return int(self.linear.out_features)

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

    @property
    def action_dim(self) -> int:
        """Dimensionality of the Normal distribution (per-axis means/scales)."""
        return int(self.mu_linear.out_features)

    def __call__(self, features: Float[Array, " d"]) -> Normal:
        """Map features to a diagonal Normal distribution."""
        mu = self.mu_linear(features)
        log_sigma = jnp.clip(self.log_sigma_linear(features), self.log_sigma_min, self.log_sigma_max)
        return Normal(loc=mu, scale=jnp.exp(log_sigma))


class SquashedGaussianHead(eqx.Module):
    r"""Maps features to a tanh-squashed Normal distribution.

    Used by SAC for bounded continuous actions.  Returns a
    :class:`rltrain.distributions.SquashedNormal` with a numerically stable
    ``log_prob`` that avoids the catastrophic cancellation in distreqx's
    ``Transformed(Normal, Tanh)`` bijector at saturation.
    """

    gaussian: GaussianHead

    def __init__(self, feature_dim: int, action_dim: int, *, key: PRNGKeyArray):
        """Initialise the underlying Gaussian head."""
        self.gaussian = GaussianHead(feature_dim, action_dim, key=key)

    @property
    def action_dim(self) -> int:
        """Forwarded from the wrapped Gaussian head."""
        return self.gaussian.action_dim

    def __call__(self, features: Float[Array, " d"]):
        """Map features to a numerically stable squashed Normal distribution."""
        from rltrain.distributions import SquashedNormal

        base_dist = self.gaussian(features)
        return SquashedNormal(loc=base_dist.loc, scale=base_dist.scale)


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

    @property
    def action_dim(self) -> int:
        """Dimensionality of the Gamma distribution."""
        return int(self.alpha_linear.out_features)

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

    @property
    def action_dim(self) -> int:
        """Dimensionality of the Beta distribution."""
        return int(self.alpha_linear.out_features)

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
    ``rltrain.math.project_distribution`` and ``rltrain.math.q_values_from_pmf``.
    """

    linear: eqx.nn.Linear
    num_atoms: int = eqx.field(static=True)
    num_actions: int = eqx.field(static=True)
    v_min: float = eqx.field(static=True)
    v_max: float = eqx.field(static=True)

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
        self.num_atoms = num_atoms
        self.num_actions = num_actions
        self.v_min = v_min
        self.v_max = v_max

    @property
    def atoms(self) -> Float[Array, " num_atoms"]:
        """Fixed atom support vector, computed from static v_min/v_max/num_atoms."""
        return jnp.linspace(self.v_min, self.v_max, self.num_atoms)

    def __call__(self, features: Float[Array, " d"]) -> Float[Array, "num_actions num_atoms"]:
        """Map features to per-action PMFs via softmax over the atom axis."""
        logits = self.linear(features)
        logits = logits.reshape(self.num_actions, self.num_atoms)
        return jax.nn.softmax(logits, axis=-1)
