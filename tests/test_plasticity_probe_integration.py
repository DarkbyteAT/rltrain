"""Integration test for the example-local PlasticityProbeCallback.

Verifies that a 100-step live SAC training run with the
:class:`PlasticityProbeCallback` wired in produces a ``probes.csv`` with at
least one row and valid headers. The callback lives under
``examples/live_sac_plasticity/`` rather than ``rltrain/callbacks/`` because
the staff-architect's ``FeatureExtractor`` Protocol is a follow-up; this test
asserts the example-local contract holds end-to-end against a real SAC agent.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import pytest

from rltrain.agents.sac import SAC
from rltrain.env import GymnaxEnv
from rltrain.heads import DiscreteHead
from rltrain.networks import ConvD2RLMLP
from rltrain.trainer import Trainer


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from examples.live_sac_plasticity.plasticity_probe import (  # noqa: E402 — path-injected import
    PlasticityProbeCallback,
    fixed_random_obs_provider,
)


@pytest.mark.integration
def test_plasticity_probe_callback_writes_csv_during_live_training(tmp_path):
    """Given a 100-step SAC run on Breakout-MinAtar with the probe callback
    attached, ``probes.csv`` exists with header + at least one row."""
    # Given: a SAC agent on Breakout-MinAtar with a ConvD2RLMLP backbone.
    key = jax.random.key(0)
    k_net, k_obs, k_fit = jax.random.split(key, 3)
    k_actor, k_c1, k_c2, k_head = jax.random.split(k_net, 4)

    env = GymnaxEnv("Breakout-MinAtar", num_envs=4)

    actor = ConvD2RLMLP(
        height=10,
        width=10,
        in_channels=4,
        out_size=128,
        conv_channels=16,
        conv_kernel=3,
        feature_dim=128,
        mlp_width=64,
        mlp_depth=2,
        key=k_actor,
    )
    critic_1 = ConvD2RLMLP(
        height=10,
        width=10,
        in_channels=4,
        out_size=3,
        conv_channels=16,
        conv_kernel=3,
        feature_dim=128,
        mlp_width=64,
        mlp_depth=2,
        key=k_c1,
    )
    critic_2 = ConvD2RLMLP(
        height=10,
        width=10,
        in_channels=4,
        out_size=3,
        conv_channels=16,
        conv_kernel=3,
        feature_dim=128,
        mlp_width=64,
        mlp_depth=2,
        key=k_c2,
    )
    head = DiscreteHead(feature_dim=128, action_dim=3, key=k_head)

    agent = SAC(
        actor=actor,
        action_head=head,
        critic_1=critic_1,
        critic_2=critic_2,
        actor_optimizer=optax.adam(3e-4),
        critic_optimizer=optax.adam(3e-4),
        alpha_optimizer=optax.adam(1e-4),
        gamma=0.99,
        tau=0.01,
        target_entropy=1.0768,
    )

    obs_provider = fixed_random_obs_provider(env, k_obs, batch_size=64)
    probe_cb = PlasticityProbeCallback(agent=agent, obs_provider=obs_provider, cadence_steps=0)

    run_dir = tmp_path / "live_probe_smoke"
    run_dir.mkdir()

    trainer = Trainer(
        agent,
        env,
        num_steps=100,
        checkpoint_steps=50,
        run_dir=run_dir,
        batch_size=32,
        buffer_capacity=2_000,
        min_buffer_size=64,
        prioritised=True,
        callbacks=[probe_cb],
        seed=0,
    )

    # When: a 100-step run completes.
    trainer.fit(k_fit)

    # Then: probes.csv exists with the expected header and at least one row.
    probes_csv = run_dir / "probes.csv"
    assert probes_csv.exists(), f"probes.csv was not written under {run_dir}"
    with probes_csv.open() as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["step", "effective_rank", "sign_entropy"], f"Unexpected header: {rows[0]!r}"
    assert len(rows) >= 2, f"Expected at least one probe row, got {len(rows) - 1}"
    # And every probe row carries finite numeric values.
    for row in rows[1:]:
        step = int(row[0])
        eff_rank = float(row[1])
        sign_ent = float(row[2])
        assert step >= 0
        assert jnp.isfinite(eff_rank), f"effective_rank is non-finite at step {step}"
        assert jnp.isfinite(sign_ent), f"sign_entropy is non-finite at step {step}"
