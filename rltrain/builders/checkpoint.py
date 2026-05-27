"""Checkpoint loading — restore a trained agent state from a run directory."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

from rltrain.agents.agent import Agent
from rltrain.builders.agent import agent as build_agent


log = logging.getLogger(__name__)


def load_agent(
    run_dir: str | Path,
    *,
    checkpoint: str = "FINAL",
    key: PRNGKeyArray | None = None,
) -> tuple[Agent, Any]:
    """Load a trained agent and its serialised training state.

    Reconstructs the agent module from ``config/agent.json``, builds a fresh
    ``TrainState`` template, then overlays the serialised leaves from
    ``models/model_{checkpoint}.eqx``.

    Args:
        run_dir: Run directory containing ``config/`` and ``models/``.
        checkpoint: ``"FINAL"`` for the final checkpoint, or a step string.
        key: Optional base PRNG key for re-initialising the agent module.
            Defaults to ``PRNGKey(0)``.

    Returns:
        ``(agent, state)`` — the agent module and the deserialised training
        state. Use ``agent.act(state, obs, key)`` for inference.
    """
    run_dir = Path(run_dir)
    if not (checkpoint == "FINAL" or checkpoint.isdigit()):
        raise ValueError(f"checkpoint must be 'FINAL' or a numeric step, got {checkpoint!r}")

    cfg_path = run_dir / "config" / "agent.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Agent config not found: {cfg_path}")
    cfg = json.loads(cfg_path.read_text())

    if key is None:
        key = jax.random.PRNGKey(0)
    agent_module = build_agent(key=key, **cfg)

    template_state = agent_module.init(key)
    model_path = run_dir / "models" / f"model_{checkpoint}.eqx"
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    state = eqx.tree_deserialise_leaves(str(model_path), template_state)
    log.info("loaded agent from '%s' (checkpoint=%s)", run_dir, checkpoint)
    return agent_module, state
