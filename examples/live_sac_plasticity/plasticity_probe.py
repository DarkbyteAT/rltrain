"""Plasticity probe callback — logs effective rank + sign entropy at checkpoints.

Probes the actor's bottleneck features on a fixed obs batch at each checkpoint
and appends a row to ``probes.csv``. The callback is architecture-agnostic by
contract: it calls ``actor.bottleneck_features(obs)`` and lets the architecture
decide what that means. Both :class:`rltrain.networks.ConvD2RLMLP` and
:class:`rltrain.networks.ConvFourierD2RLMLP` implement the method with matching
shape (``(feature_dim,)`` single-sample, ``(batch, feature_dim)`` under vmap),
so swapping arch requires no callback edit.

Lives in ``examples/live_sac_plasticity/`` rather than ``rltrain/callbacks/``
because the staff-architect's ``FeatureExtractor`` Protocol — the proper home
for this — is a follow-up. Until then, this example-local callback is the
explicit contract: an ``agent`` whose actor exposes ``.bottleneck_features``
and an ``obs_provider`` callable returning a probe batch.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from rltrain.networks import fourier_feature_effective_rank, fourier_feature_sign_entropy


logger = logging.getLogger(__name__)


class PlasticityProbeCallback:
    """Logs effective rank + sign entropy of the actor bottleneck at checkpoints.

    Args:
        agent: The SAC agent module (static ``eqx.Module``). Its
            ``_reconstruct_actor(state.actor_params)`` is used at probe time
            to combine the checkpoint params with the static actor structure.
        obs_provider: Zero-arg callable returning a ``(batch, *obs_shape)``
            JAX array of observations to probe. The runner is responsible for
            sourcing these (e.g. a frozen eval batch, or a getter into the
            replay buffer); the callback only knows the contract.
        cadence_steps: Probe every ``cadence_steps`` between checkpoints.
            ``0`` (the default) probes at every checkpoint.
    """

    def __init__(
        self,
        agent,
        obs_provider: Callable[[], Float[Array, "B ..."]],
        cadence_steps: int = 0,
    ) -> None:
        """Capture agent reference, obs provider, and cadence."""
        self._agent = agent
        self._obs_provider = obs_provider
        self._cadence_steps = cadence_steps
        self._probes_path: Path | None = None
        self._writer: csv.writer | None = None
        self._file = None
        self._last_logged_step: int = -1

    def on_train_start(self, config: dict, run_dir: Path | None) -> None:
        """Open ``probes.csv`` and write the header."""
        if run_dir is None:
            return
        self._probes_path = Path(run_dir) / "probes.csv"
        self._file = open(self._probes_path, "w", newline="")  # noqa: SIM115 — closed in on_train_end
        self._writer = csv.writer(self._file)
        self._writer.writerow(["step", "effective_rank", "sign_entropy"])
        self._file.flush()

    def on_step(self, step: int, metrics: dict[str, float]) -> None:
        """No-op — probes are checkpoint-driven."""

    def on_episode_end(
        self,
        episode: int,
        episode_return: float,
        episode_length: int,
        running_return: float = 0.0,
    ) -> None:
        """No-op — probes are checkpoint-driven."""

    def on_checkpoint(self, step: int, agent_state, run_dir: Path | None) -> None:
        """Compute effective rank + sign entropy and append a row to ``probes.csv``."""
        if self._writer is None or self._file is None:
            return
        if self._cadence_steps > 0 and step - self._last_logged_step < self._cadence_steps:
            return

        obs_batch = self._obs_provider()
        actor, _head = self._agent._reconstruct_actor(agent_state.actor_params)
        feats = jax.vmap(actor.bottleneck_features)(obs_batch)
        # Diagnostics are JAX scalars; cast to float before writing.
        eff_rank = float(fourier_feature_effective_rank(feats))
        sign_ent = float(fourier_feature_sign_entropy(feats))

        self._writer.writerow([step, eff_rank, sign_ent])
        self._file.flush()
        self._last_logged_step = step
        logger.info(
            "probe step=%d effective_rank=%.3f sign_entropy=%.3f",
            step,
            eff_rank,
            sign_ent,
        )

    def on_train_end(self, agent_state, run_dir: Path | None) -> None:
        """Close the probes file."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


def fixed_random_obs_provider(env, key, batch_size: int = 256) -> Callable[[], Float[Array, "B ..."]]:
    """Build an ``obs_provider`` that returns a frozen batch of random-policy obs.

    Rolls the env under a uniform random policy for ``batch_size`` steps, collects
    the visited observations, and returns a closure that hands them back
    verbatim. The same batch is reused at every probe call, so rank/entropy
    deltas across steps reflect changes in the network's response to a fixed
    stimulus — not changes in the stimulus.

    Args:
        env: A vectorised ``GymnaxEnv`` (or compatible) with ``num_envs``,
            ``reset(key) → state``, ``step(state, action, key) → state``,
            and ``num_actions``.
        key: PRNG key seeding the obs collection.
        batch_size: Number of obs frames to collect. The provider returns
            a ``(batch_size, *obs_shape)`` array.

    Returns:
        A zero-arg callable yielding the frozen obs batch.
    """
    obs_frames = []
    state = env.reset(key)
    steps_needed = (batch_size + env.num_envs - 1) // env.num_envs
    for _ in range(steps_needed):
        key, k_act, k_step = jax.random.split(key, 3)
        action = jax.random.randint(k_act, shape=(env.num_envs,), minval=0, maxval=env.num_actions)
        obs_frames.append(jnp.asarray(state.obs))
        state = env.step(state, action, k_step)
    # Stack along leading axis: (steps_needed, num_envs, *obs_shape) -> flatten -> trim.
    stacked = jnp.concatenate(obs_frames, axis=0)
    frozen = stacked[:batch_size]

    def _provider() -> Float[Array, "B ..."]:
        return frozen

    return _provider
