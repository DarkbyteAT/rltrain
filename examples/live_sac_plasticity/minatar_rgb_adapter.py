"""Gymnasium-shaped RGB adapter around gymnax MinAtar envs.

gymnax provides MinAtar envs as pure JAX pure-step functions; gymnasium has no
MinAtar namespace registered out of the box. The :class:`VideoRecorderCallback`
expects a gymnasium-shaped env with ``reset()``, ``step()``, ``render()``, and
``render_mode = "rgb_array"``. This adapter wraps a gymnax MinAtar env in that
shape, plus a channel-to-colour upsampled render.

MinAtar observations are HWC binary masks (one channel per game object class).
For ``render()`` we assign each channel a distinct colour and blend onto a
black background. Upsampling produces a watchable MP4 without needing a real
MinAtar renderer.

Single-env only — this is for evaluation rollouts, not training.
"""

from __future__ import annotations

import jax
import numpy as np
from gymnax.environments.minatar import (
    asterix,
    breakout,
    freeway,
    seaquest,
    space_invaders,
)


_ENV_REGISTRY = {
    "Asterix-MinAtar": asterix.MinAsterix,
    "Breakout-MinAtar": breakout.MinBreakout,
    "Freeway-MinAtar": freeway.MinFreeway,
    "Seaquest-MinAtar": seaquest.MinSeaquest,
    "SpaceInvaders-MinAtar": space_invaders.MinSpaceInvaders,
}


# Distinct RGB colours per object class. Cycles past channel 7 if needed; the
# MinAtar games we wrap top out at 6 channels (SpaceInvaders).
_CHANNEL_COLOURS = np.array(
    [
        [255, 80, 80],  # red — channel 0 (typically the player / paddle)
        [80, 200, 255],  # cyan
        [255, 200, 80],  # amber
        [180, 80, 255],  # purple
        [80, 255, 120],  # green
        [255, 255, 255],  # white
        [255, 80, 200],  # magenta
        [200, 200, 200],  # grey (fallback)
    ],
    dtype=np.uint8,
)


class _GymnaxMinAtarRGBAdapter:
    """gymnasium-style single-env wrapper around a gymnax MinAtar env.

    Exposes ``reset()``, ``step()``, ``render()``, and ``close()`` matching
    the gymnasium API surface that :class:`VideoRecorderCallback` consumes.
    Observations come straight from gymnax (HWC binary mask); ``render()``
    colour-maps and upsamples that mask to an RGB frame.

    Args:
        env_id: MinAtar env id (e.g. ``"Breakout-MinAtar"``).
        render_scale: Pixel-replication factor for the rendered frame so the
            10x10 obs becomes a watchable resolution. Default 16 -> 160x160.
        seed: Seed for the adapter's internal PRNG key. The eval loop calls
            ``reset()`` with no arg, so the adapter manages its own key.
    """

    metadata = {"render_modes": ["rgb_array"]}
    render_mode = "rgb_array"

    def __init__(self, env_id: str, *, render_scale: int = 16, seed: int = 0) -> None:
        """Construct the gymnax env and seed the internal PRNG."""
        if env_id not in _ENV_REGISTRY:
            raise ValueError(f"unknown MinAtar env_id={env_id!r}; expected one of {list(_ENV_REGISTRY)}")
        self._env = _ENV_REGISTRY[env_id]()
        self._env_params = self._env.default_params
        self._render_scale = render_scale
        self._key = jax.random.key(seed)
        self._state = None
        self._last_obs = None
        self._reset_fn = jax.jit(self._env.reset)
        self._step_fn = jax.jit(self._env.step)

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """Reset the env and return ``(obs, info)`` in gymnasium format."""
        if seed is not None:
            self._key = jax.random.key(seed)
        self._key, k_reset = jax.random.split(self._key)
        obs, state = self._reset_fn(k_reset, self._env_params)
        self._state = state
        self._last_obs = np.asarray(obs)
        return self._last_obs, {}

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Step the env and return ``(obs, reward, terminated, truncated, info)``."""
        self._key, k_step = jax.random.split(self._key)
        obs, state, reward, done, _info = self._step_fn(k_step, self._state, int(action), self._env_params)
        self._state = state
        self._last_obs = np.asarray(obs)
        return self._last_obs, float(reward), bool(done), False, {}

    def render(self) -> np.ndarray:
        """Render the current obs as an upsampled RGB frame."""
        if self._last_obs is None:
            return np.zeros((10 * self._render_scale, 10 * self._render_scale, 3), dtype=np.uint8)
        return _channel_mask_to_rgb(self._last_obs, scale=self._render_scale)

    def close(self) -> None:
        """No-op — gymnax envs are pure functions, nothing to release."""


def _channel_mask_to_rgb(obs_hwc: np.ndarray, *, scale: int = 16) -> np.ndarray:
    """Blend an HWC binary-channel obs onto a black RGB canvas, upsampled.

    Where multiple channels overlap on the same cell, later channels paint
    over earlier ones. ``scale`` replicates each obs cell into a ``scale x
    scale`` block of pixels so the rendered frame is at a watchable size.
    """
    h, w, c = obs_hwc.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for ch in range(c):
        mask = obs_hwc[..., ch] > 0.5
        canvas[mask] = _CHANNEL_COLOURS[ch % len(_CHANNEL_COLOURS)]
    if scale > 1:
        canvas = np.kron(canvas, np.ones((scale, scale, 1), dtype=np.uint8))
    return canvas


# Sanity-import the helper so the adapter's render() path is exercised even if
# the eval-rollout VideoRecorderCallback path is gated off in some smokes.
__all__ = ["_GymnaxMinAtarRGBAdapter", "_channel_mask_to_rgb"]
