r"""Numerically stable probability distributions for RL.

Provides ``SquashedNormal`` — a tanh-squashed Gaussian with a stable
``log_prob`` that avoids the catastrophic cancellation in distreqx's
``Transformed(Normal, Tanh)`` bijector.

The standard correction term $\log(1 - \tanh^2(x))$ underflows to
$-\infty$ when $|\tanh(x)| \approx 1$.  We use the algebraic identity

$$\log(1 - \tanh^2(x)) = 2\,(\log 2 - x - \mathrm{softplus}(-2x))$$

which is numerically stable for all $x$.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class SquashedNormal(eqx.Module):
    r"""Tanh-squashed Normal distribution with numerically stable log_prob.

    Samples $a = \tanh(u)$ where $u \sim \mathcal{N}(\mu, \sigma^2)$.
    The log-probability corrects for the change of variables:

    $$\log \pi(a) = \log \mathcal{N}(u; \mu, \sigma)
                   - \sum_i \log(1 - a_i^2)$$

    using the stable form $\log(1 - \tanh^2(x)) = 2(\log 2 - x - \mathrm{softplus}(-2x))$.

    Attributes:
        loc: Mean $\mu$ of the unsquashed Normal.
        scale: Standard deviation $\sigma$ of the unsquashed Normal.
    """

    loc: Float[Array, " *d"]
    scale: Float[Array, " *d"]

    def sample(self, key: PRNGKeyArray) -> Float[Array, " *d"]:
        r"""Sample $a = \tanh(u)$ where $u \sim \mathcal{N}(\mu, \sigma)$."""
        u = self.loc + self.scale * jax.random.normal(key, self.loc.shape)
        return jnp.tanh(u)

    def sample_and_log_prob(self, key: PRNGKeyArray) -> tuple[Float[Array, " *d"], Float[Array, " *d"]]:
        """Sample and compute log_prob in one pass (avoids atanh)."""
        u = self.loc + self.scale * jax.random.normal(key, self.loc.shape)
        a = jnp.tanh(u)
        log_prob = self._log_prob_from_unsquashed(u)
        return a, log_prob

    def log_prob(self, action: Float[Array, " *d"]) -> Float[Array, " *d"]:
        r"""Compute $\log \pi(a)$ with numerically stable Jacobian correction.

        Inverts the tanh via ``atanh`` (which is stable for $|a| < 1$)
        then applies the correction term.
        """
        # Clamp to avoid atanh(±1) = ±inf
        u = jnp.arctanh(jnp.clip(action, -1.0 + 1e-6, 1.0 - 1e-6))
        return self._log_prob_from_unsquashed(u)

    def entropy(self) -> Float[Array, " *d"]:
        """Upper bound on the entropy of the squashed distribution.

        Returns the entropy of the unsquashed Normal, which is strictly
        greater than the true entropy of the tanh-squashed distribution
        (tanh reduces support from R to (-1,1), lowering entropy).
        No closed-form expression exists for the squashed entropy.
        """
        return 0.5 * jnp.log(2.0 * jnp.pi * jnp.e * self.scale**2)

    def _log_prob_from_unsquashed(self, u: Float[Array, " *d"]) -> Float[Array, " *d"]:
        r"""Compute log_prob given the pre-tanh sample $u$.

        $$\log \pi(a) = -\frac{(u - \mu)^2}{2\sigma^2} - \log\sigma
                       - \frac{1}{2}\log(2\pi)
                       - \sum_i 2(\log 2 - u_i - \mathrm{softplus}(-2u_i))$$
        """
        # Normal log_prob
        var = self.scale**2
        normal_lp = -0.5 * ((u - self.loc) ** 2 / var + jnp.log(var) + jnp.log(2.0 * jnp.pi))

        # Stable Jacobian correction: log(1 - tanh²(u))
        # = log(sech²(u)) = 2·(log(2) - u - softplus(-2u))
        correction = 2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))

        return normal_lp - correction
