"""Checkpoint loading — reconstruct trained agents from saved state dicts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch as T

import rltrain.utils.builders as mk
from rltrain.agents import Agent
from rltrain.utils.device import resolve_device


log = logging.getLogger(__name__)


def load_agent(
    run_dir: str | Path,
    *,
    checkpoint: str = "FINAL",
    device: str | T.device = "auto",
) -> Agent:
    """Load a trained agent from a run directory.

    Reconstructs the agent's network architecture from the saved config,
    loads the checkpoint weights, and sets the model to evaluation mode.
    The returned agent is ready for inference (calling ``agent(states)``
    to sample actions) but **not** for continued training — optimizer
    state and replay buffers are not restored.

    Parameters
    ----------
    ``run_dir``
        Path to the run directory (the one containing ``config/`` and
        ``models/`` subdirectories).
    ``checkpoint``
        Which checkpoint to load. ``"FINAL"`` loads ``model_FINAL.pt``;
        an integer string like ``"2500"`` loads ``model_2500.pt``.
    ``device``
        Device to place the agent on. Accepts any string recognised by
        ``resolve_device`` (``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``)
        or a ``torch.device`` instance.

    Returns
    -------
    ``Agent``
        The loaded agent in evaluation mode.

    Raises
    ------
    ``FileNotFoundError``
        If the agent config or checkpoint file does not exist.
    ``ValueError``
        If ``checkpoint`` is not ``"FINAL"`` or a numeric string.
    """
    run_dir = Path(run_dir)

    # --- validate checkpoint name ---------------------------------------- #
    if not (checkpoint == "FINAL" or checkpoint.isdigit()):
        raise ValueError(f"checkpoint must be 'FINAL' or a numeric step, got {checkpoint!r}")

    # --- resolve device -------------------------------------------------- #
    if isinstance(device, str):
        device = resolve_device(device)

    # --- load agent config ----------------------------------------------- #
    config_path = run_dir / "config" / "agent.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Agent config not found: {config_path}")

    with open(config_path) as f:
        cfg = json.load(f)

    # --- reconstruct agent via builder ----------------------------------- #
    agent = mk.agent(device=device, **cfg)
    agent.setup()

    # --- load checkpoint weights ----------------------------------------- #
    model_path = run_dir / "models" / f"model_{checkpoint}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    state_dict = T.load(model_path, map_location=device, weights_only=True)
    agent.model.load_state_dict(state_dict)
    agent.model.eval()

    # --- DQN-specific post-load fixups ----------------------------------- #
    if hasattr(agent, "target"):
        agent.target.load_state_dict(agent.model["qnet"].state_dict())
    if hasattr(agent, "eps_greedy"):
        agent.eps_greedy = 0.0

    log.info("loaded agent from '%s' (checkpoint=%s, device=%s)", run_dir, checkpoint, device)
    return agent
