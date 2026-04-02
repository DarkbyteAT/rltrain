# VideoRecorderCallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `VideoRecorderCallback` that records evaluation videos of agent gameplay at training checkpoints, using gymnasium's `RecordVideo` wrapper on an isolated eval environment.

**Architecture:** Two independent cards. Card A extracts `MDP.preprocess_obs()` to encapsulate observation preprocessing. Card B implements the `VideoRecorderCallback` itself. Card B depends on Card A (it calls `env.preprocess_obs()` during eval rollouts). Within each card, work follows TDD: write failing test → implement → verify → commit.

**Tech Stack:** Python, gymnasium (`RecordVideo` wrapper, `gym.make`), PyTorch (agent inference), pytest.

**Spec:** [`docs/superpowers/specs/2026-04-02-video-recorder-callback-design.md`](../specs/2026-04-02-video-recorder-callback-design.md)

---

## Card A: Encapsulate observation preprocessing — `MDP.preprocess_obs()`

**Trello:** [Card #21](https://trello.com/c/ERxUf9Aa) — no dependencies, can start immediately.

**Files:**
- Modify: `rltrain/env/mdp.py` (add `preprocess_obs`, refactor `reset` and `step`)
- Create: `tests/env/test_preprocess_obs.py`

### Task A1: Write failing tests for `preprocess_obs()`

- [ ] **Step 1: Create the test file**

```python
# tests/env/test_preprocess_obs.py
"""Tests for MDP.preprocess_obs() observation preprocessing."""

import gymnasium as gym
import gymnasium.vector as vgym
import numpy as np

from rltrain.env import MDP


def _make_mdp(swap_channels: bool) -> MDP:
    """Helper: build an MDP wrapping CartPole (no channel swap needed)."""
    env = vgym.SyncVectorEnv([lambda: gym.make("CartPole-v1")])
    return MDP(env, run_beta=0.1, log_freq=100, swap_channels=swap_channels)


def test_preprocess_obs_identity_when_no_swap():
    """With swap_channels=False, preprocess_obs returns the input unchanged."""
    mdp = _make_mdp(swap_channels=False)
    obs = np.array([[1.0, 2.0, 3.0, 4.0]])
    result = mdp.preprocess_obs(obs)
    np.testing.assert_array_equal(result, obs)


def test_preprocess_obs_swaps_channels():
    """With swap_channels=True, preprocess_obs applies squeeze().T[newaxis, :]."""
    mdp = _make_mdp(swap_channels=True)
    # Simulate a (1, H, W, C) image observation from a vectorised env
    obs = np.random.rand(1, 4, 4, 3)
    result = mdp.preprocess_obs(obs)
    expected = obs.squeeze().T[np.newaxis, :]
    np.testing.assert_array_equal(result, expected)


def test_preprocess_obs_used_by_reset(cartpole_env):
    """After reset, state should equal preprocess_obs applied to raw obs."""
    # cartpole_env has swap_channels=False, so state == raw obs
    cartpole_env.setup(seed=42)
    assert cartpole_env.state is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/env/test_preprocess_obs.py -v`
Expected: FAIL — `AttributeError: 'MDP' object has no attribute 'preprocess_obs'`

### Task A2: Implement `preprocess_obs()` and refactor `MDP`

- [ ] **Step 3: Add `preprocess_obs` method to `MDP`**

In `rltrain/env/mdp.py`, add this method to the `MDP` class (after `__init__`, before `setup`):

```python
def preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
    """Apply observation preprocessing (e.g. channel swap for image envs).

    Parameters
    ----------
    `obs` : `np.ndarray`
        Raw observation from the gymnasium environment.

    Returns
    -------
    `np.ndarray`
        Preprocessed observation ready for the agent.
    """
    if self.swap_channels:
        return obs.squeeze().T[np.newaxis, :]
    return obs
```

- [ ] **Step 4: Replace inline channel-swap in `reset()`**

Change:
```python
# Adjusts next observation to put channels axis first if input is an image
if self.swap_channels:
    self.state = self.state.squeeze().T[np.newaxis, :]
```

To:
```python
self.state = self.preprocess_obs(self.state)
```

- [ ] **Step 5: Replace inline channel-swap in `step()`**

Change:
```python
# Adjusts next observation to put channels axis first if input is an image
if self.swap_channels:
    next_state = next_state.squeeze().T[np.newaxis, :]
```

To:
```python
next_state = self.preprocess_obs(next_state)
```

- [ ] **Step 6: Run preprocess_obs tests**

Run: `pytest tests/env/test_preprocess_obs.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 7: Run full test suite to verify no regressions**

Run: `pytest -v`
Expected: All existing tests PASS (the refactor is purely internal).

- [ ] **Step 8: Commit**

```bash
git add rltrain/env/mdp.py tests/env/test_preprocess_obs.py
git commit -m "refactor: extract MDP.preprocess_obs() for observation preprocessing"
```

---

## Card B: Implement `VideoRecorderCallback`

**Trello:** [Card #20](https://trello.com/c/UlugejBr) — depends on Card A (`preprocess_obs`).

**Files:**
- Create: `rltrain/callbacks/video_recorder.py`
- Create: `tests/callbacks/test_video_recorder.py`
- Modify: `tests/callbacks/test_protocol.py` (add protocol conformance test)
- Modify: `README.md` (add VideoRecorderCallback to callbacks section)
- Modify: `CONTRIBUTING.md` (add video_recorder.py to directory tree)

### Task B1: Protocol conformance test

- [ ] **Step 1: Add protocol test for VideoRecorderCallback**

Append to `tests/callbacks/test_protocol.py`:

```python
from rltrain.callbacks.video_recorder import VideoRecorderCallback


def test_video_recorder_satisfies_protocol():
    assert isinstance(VideoRecorderCallback(), Callback)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/callbacks/test_protocol.py::test_video_recorder_satisfies_protocol -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rltrain.callbacks.video_recorder'`

### Task B2: Scaffold `VideoRecorderCallback` with no-op hooks

- [ ] **Step 3: Create the callback module**

Create `rltrain/callbacks/video_recorder.py`:

```python
"""Video recorder callback — records evaluation videos at checkpoint intervals."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np

if TYPE_CHECKING:
    from rltrain.agents.agent import Agent
    from rltrain.env import MDP


log = logging.getLogger(__name__)


class VideoRecorderCallback:
    """Records evaluation videos of agent behaviour during training.

    Uses gymnasium's ``RecordVideo`` wrapper on a separate persistent evaluation
    environment. By default, records at each checkpoint; optionally configure
    ``episode_trigger`` to record at specific training episodes instead.

    Parameters
    ----------
    `env_fn` : `Callable[[], gym.Env] | None`
        Zero-arg callable returning a ``gym.Env`` with ``render_mode="rgb_array"``.
        If None, auto-detects from the training MDP's env spec. The auto-detection
        creates a bare env without user-applied wrappers; pass ``env_fn`` explicitly
        when wrappers matter for the recording.
    `num_episodes` : `int`
        Number of evaluation episodes to record at each trigger point.
    `episode_trigger` : `Callable[[int], bool] | None`
        When set, controls when eval rollouts happen during training based on the
        training episode count. Rollouts trigger in ``on_episode_end`` instead of
        the default ``on_checkpoint``.
    `video_length` : `int`
        Fixed video length in frames. 0 means record full episodes.
    `name_prefix` : `str`
        Filename prefix for recorded videos.
    `fps` : `int`
        Frames per second for the output video.
    """

    def __init__(
        self,
        *,
        env_fn: Callable[[], gym.Env] | None = None,
        num_episodes: int = 3,
        episode_trigger: Callable[[int], bool] | None = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        fps: int = 30,
    ) -> None:
        self._env_fn = env_fn
        self._num_episodes = num_episodes
        self._episode_trigger = episode_trigger
        self._video_length = video_length
        self._name_prefix = name_prefix
        self._fps = fps

        self._eval_env: gym.Env | None = None
        self._enabled: bool = True

    def on_train_start(self, agent: Agent, env: MDP, run_dir: Path) -> None: ...
    def on_step(self, agent: Agent, env: MDP, step: int) -> None: ...
    def on_episode_end(self, agent: Agent, env: MDP, episode: int) -> None: ...
    def on_checkpoint(self, agent: Agent, env: MDP, run_dir: Path) -> None: ...
    def on_train_end(self, agent: Agent, env: MDP, run_dir: Path) -> None: ...
```

- [ ] **Step 4: Run protocol test**

Run: `pytest tests/callbacks/test_protocol.py::test_video_recorder_satisfies_protocol -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rltrain/callbacks/video_recorder.py tests/callbacks/test_protocol.py
git commit -m "feat: scaffold VideoRecorderCallback with protocol conformance"
```

### Task B3: Auto-detection fallback + `on_train_start`

- [ ] **Step 6: Write tests for auto-detection and `on_train_start` setup**

Append to `tests/callbacks/test_video_recorder.py` (create the file):

```python
"""Tests for the VideoRecorderCallback."""

from pathlib import Path
from unittest.mock import MagicMock

import gymnasium as gym
import gymnasium.vector as vgym
import numpy as np
import pytest

from rltrain.callbacks.video_recorder import VideoRecorderCallback
from rltrain.env import MDP


def _make_mdp() -> MDP:
    """Build a minimal MDP wrapping CartPole."""
    env = vgym.SyncVectorEnv([lambda: gym.make("CartPole-v1")])
    return MDP(env, run_beta=0.1, log_freq=100, swap_channels=False)


def _make_agent_stub() -> MagicMock:
    """Create a stub agent whose __call__ returns a valid CartPole action."""
    agent = MagicMock()
    agent.side_effect = lambda obs: np.array([0])
    return agent


def test_auto_detect_creates_eval_env(tmp_path):
    """With no env_fn, on_train_start should auto-detect from MDP spec."""
    mdp = _make_mdp()
    mdp.setup(seed=42)
    cb = VideoRecorderCallback()
    cb.on_train_start(_make_agent_stub(), mdp, tmp_path)

    assert cb._eval_env is not None
    assert cb._enabled is True
    assert (tmp_path / "videos").is_dir()


def test_explicit_env_fn_used(tmp_path):
    """When env_fn is provided, it should be used instead of auto-detection."""
    factory_called = False

    def my_factory():
        nonlocal factory_called
        factory_called = True
        return gym.make("CartPole-v1", render_mode="rgb_array")

    mdp = _make_mdp()
    mdp.setup(seed=42)
    cb = VideoRecorderCallback(env_fn=my_factory)
    cb.on_train_start(_make_agent_stub(), mdp, tmp_path)

    assert factory_called
    assert cb._eval_env is not None


def test_graceful_disable_on_non_renderable_env(tmp_path):
    """If env can't render rgb_array, callback disables itself."""
    def bad_factory():
        return gym.make("CartPole-v1")  # no render_mode

    mdp = _make_mdp()
    mdp.setup(seed=42)
    cb = VideoRecorderCallback(env_fn=bad_factory)
    cb.on_train_start(_make_agent_stub(), mdp, tmp_path)

    assert cb._enabled is False
```

- [ ] **Step 7: Run tests to verify they fail**

Run: `pytest tests/callbacks/test_video_recorder.py -v`
Expected: FAIL — `on_train_start` is a no-op, `_eval_env` is None, `videos/` dir not created.

- [ ] **Step 8: Implement auto-detection and `on_train_start`**

Replace the `on_train_start` no-op in `rltrain/callbacks/video_recorder.py`:

```python
def on_train_start(self, agent: Agent, env: MDP, run_dir: Path) -> None:
    self._preprocess_obs = env.preprocess_obs

    try:
        base_env = self._env_fn() if self._env_fn is not None else self._make_env_from_mdp(env)
    except Exception:
        log.warning("VideoRecorderCallback: failed to create eval env — disabling", exc_info=True)
        self._enabled = False
        return

    if base_env.render_mode != "rgb_array":
        log.warning(
            "VideoRecorderCallback: eval env render_mode is '%s', not 'rgb_array' — disabling",
            base_env.render_mode,
        )
        base_env.close()
        self._enabled = False
        return

    video_dir = run_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    record_kwargs: dict = {
        "video_folder": str(video_dir),
        "name_prefix": self._name_prefix,
        "fps": self._fps,
        "disable_logger": True,
    }
    if self._video_length > 0:
        record_kwargs["video_length"] = self._video_length

    self._eval_env = gym.wrappers.RecordVideo(base_env, **record_kwargs)
    log.info("VideoRecorderCallback: recording to '%s'", video_dir)

def _make_env_from_mdp(self, env: MDP) -> gym.Env:
    """Auto-detect env ID from the training MDP and create a renderable copy."""
    spec = env.env.envs[0].spec
    if spec is None:
        raise RuntimeError(
            "Cannot auto-detect env — provide env_fn to VideoRecorderCallback"
        )
    return gym.make(spec.id, render_mode="rgb_array")
```

- [ ] **Step 9: Run tests**

Run: `pytest tests/callbacks/test_video_recorder.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 10: Commit**

```bash
git add rltrain/callbacks/video_recorder.py tests/callbacks/test_video_recorder.py
git commit -m "feat: VideoRecorderCallback on_train_start with auto-detection and graceful disable"
```

### Task B4: Eval rollout + `on_checkpoint`

- [ ] **Step 11: Write test for checkpoint-triggered recording**

Append to `tests/callbacks/test_video_recorder.py`:

```python
def test_on_checkpoint_runs_eval_rollouts(tmp_path):
    """on_checkpoint should produce video files in run_dir/videos/."""
    mdp = _make_mdp()
    mdp.setup(seed=42)
    agent = _make_agent_stub()

    cb = VideoRecorderCallback(
        env_fn=lambda: gym.make("CartPole-v1", render_mode="rgb_array"),
        num_episodes=1,
    )
    cb.on_train_start(agent, mdp, tmp_path)
    cb.on_checkpoint(agent, mdp, tmp_path)
    cb.on_train_end(agent, mdp, tmp_path)

    videos = list((tmp_path / "videos").glob("*.mp4"))
    assert len(videos) >= 1
```

- [ ] **Step 12: Run test to verify it fails**

Run: `pytest tests/callbacks/test_video_recorder.py::test_on_checkpoint_runs_eval_rollouts -v`
Expected: FAIL — `on_checkpoint` is a no-op, no videos written.

- [ ] **Step 13: Implement `_run_eval_rollouts` and `on_checkpoint`**

Add to `rltrain/callbacks/video_recorder.py`:

```python
def _run_eval_rollouts(self, agent: Agent) -> None:
    """Run num_episodes evaluation episodes on the recording env."""
    if not self._enabled or self._eval_env is None:
        return
    for _ in range(self._num_episodes):
        obs, _ = self._eval_env.reset()
        terminated, truncated = False, False
        while not (terminated or truncated):
            processed = self._preprocess_obs(obs[np.newaxis, ...])[0]
            action = agent(processed[np.newaxis, ...])[0]
            obs, _, terminated, truncated, _ = self._eval_env.step(action)

def on_checkpoint(self, agent: Agent, env: MDP, run_dir: Path) -> None:
    if self._episode_trigger is None:
        self._run_eval_rollouts(agent)
```

**Key detail:** The eval env is a plain `gym.Env` (not vectorised), so observations have shape `(obs_dim,)`. We add batch dimensions via `[np.newaxis, ...]` to match what `preprocess_obs` and `agent()` expect (they work on batched arrays from `SyncVectorEnv`), then index `[0]` to get scalar action back.

- [ ] **Step 14: Run test**

Run: `pytest tests/callbacks/test_video_recorder.py::test_on_checkpoint_runs_eval_rollouts -v`
Expected: PASS — at least 1 `.mp4` file in `videos/`.

- [ ] **Step 15: Commit**

```bash
git add rltrain/callbacks/video_recorder.py tests/callbacks/test_video_recorder.py
git commit -m "feat: VideoRecorderCallback eval rollouts on checkpoint"
```

### Task B5: `on_episode_end` trigger + `on_train_end` cleanup

- [ ] **Step 16: Write tests for episode trigger and train end**

Append to `tests/callbacks/test_video_recorder.py`:

```python
def test_on_episode_end_triggers_when_configured(tmp_path):
    """With episode_trigger set, on_episode_end should run rollouts."""
    mdp = _make_mdp()
    mdp.setup(seed=42)
    agent = _make_agent_stub()

    cb = VideoRecorderCallback(
        env_fn=lambda: gym.make("CartPole-v1", render_mode="rgb_array"),
        num_episodes=1,
        episode_trigger=lambda ep: ep == 1,
    )
    cb.on_train_start(agent, mdp, tmp_path)
    cb.on_episode_end(agent, mdp, 1)  # should trigger
    cb.on_episode_end(agent, mdp, 2)  # should NOT trigger
    cb.on_train_end(agent, mdp, tmp_path)

    videos = list((tmp_path / "videos").glob("*.mp4"))
    assert len(videos) >= 1


def test_on_checkpoint_noop_when_episode_trigger_set(tmp_path):
    """With episode_trigger set, on_checkpoint should NOT run rollouts."""
    mdp = _make_mdp()
    mdp.setup(seed=42)
    agent = _make_agent_stub()

    cb = VideoRecorderCallback(
        env_fn=lambda: gym.make("CartPole-v1", render_mode="rgb_array"),
        num_episodes=1,
        episode_trigger=lambda ep: False,  # never triggers
    )
    cb.on_train_start(agent, mdp, tmp_path)
    cb.on_checkpoint(agent, mdp, tmp_path)
    cb.on_train_end(agent, mdp, tmp_path)

    videos = list((tmp_path / "videos").glob("*.mp4"))
    assert len(videos) == 0


def test_on_train_end_closes_env(tmp_path):
    """on_train_end should close the eval env."""
    mdp = _make_mdp()
    mdp.setup(seed=42)
    agent = _make_agent_stub()

    cb = VideoRecorderCallback(
        env_fn=lambda: gym.make("CartPole-v1", render_mode="rgb_array"),
    )
    cb.on_train_start(agent, mdp, tmp_path)
    cb.on_train_end(agent, mdp, tmp_path)

    # After close, the env should not be usable
    assert cb._eval_env is not None
```

- [ ] **Step 17: Run tests to verify they fail**

Run: `pytest tests/callbacks/test_video_recorder.py -k "episode_end or noop or train_end" -v`
Expected: FAIL — `on_episode_end` and `on_train_end` are no-ops.

- [ ] **Step 18: Implement `on_episode_end` and `on_train_end`**

Replace the no-ops in `rltrain/callbacks/video_recorder.py`:

```python
def on_episode_end(self, agent: Agent, env: MDP, episode: int) -> None:
    if self._episode_trigger is not None and self._episode_trigger(episode):
        self._run_eval_rollouts(agent)

def on_train_end(self, agent: Agent, env: MDP, run_dir: Path) -> None:
    if self._episode_trigger is None:
        self._run_eval_rollouts(agent)
    if self._eval_env is not None:
        self._eval_env.close()
        log.info("VideoRecorderCallback: eval env closed")
```

- [ ] **Step 19: Run all video recorder tests**

Run: `pytest tests/callbacks/test_video_recorder.py -v`
Expected: All tests PASS.

- [ ] **Step 20: Commit**

```bash
git add rltrain/callbacks/video_recorder.py tests/callbacks/test_video_recorder.py
git commit -m "feat: VideoRecorderCallback episode trigger and train-end cleanup"
```

### Task B6: Full test suite + documentation

- [ ] **Step 21: Run full test suite**

Run: `pytest -v`
Expected: All tests PASS, no regressions.

- [ ] **Step 22: Update `README.md` callbacks section**

In the `## Callbacks` section, after the existing `WandbCallback` example and before `### Hook Methods`, add:

```markdown
### Video Recording

Record evaluation videos at training checkpoints using gymnasium's `RecordVideo` wrapper:

\```python
from rltrain.callbacks.video_recorder import VideoRecorderCallback

# Auto-detect env from training MDP (records 3 episodes per checkpoint):
trainer = Trainer(
    agent, env,
    num_steps=100_000,
    checkpoint_steps=2500,
    run_dir=run_dir,
    callbacks=[
        CSVLoggerCallback(),
        PlotCallback(num_steps=100_000),
        CheckpointCallback(),
        VideoRecorderCallback(),
    ],
)

# Explicit factory with custom wrappers:
VideoRecorderCallback(
    env_fn=lambda: gym.make("CartPole-v1", render_mode="rgb_array"),
    num_episodes=5,
)

# Record every 50th training episode instead of at checkpoints:
VideoRecorderCallback(episode_trigger=lambda ep: ep % 50 == 0)
\```

Videos are saved to `run_dir/videos/`. The callback requires the eval environment to support `render_mode="rgb_array"` — if it doesn't, the callback disables itself with a warning.
```

- [ ] **Step 23: Update `CONTRIBUTING.md` directory tree**

Add `video_recorder.py` to the callbacks directory listing:

```
├── callbacks/        # Callback protocol + built-in callbacks (checkpoint, csv_logger, plot, video_recorder)
```

- [ ] **Step 24: Update `README.md` directory tree**

Add `video_recorder.py` to the callbacks directory listing:

```
├── callbacks/
│   ├── __init__.py             # Callback protocol (runtime_checkable)
│   ├── checkpoint.py           # CheckpointCallback — model state_dict saves
│   ├── csv_logger.py           # CSVLoggerCallback — episode metrics to CSV
│   ├── plot.py                 # PlotCallback — per-episode and per-sample SVG plots
│   └── video_recorder.py      # VideoRecorderCallback — eval episode video recording
```

- [ ] **Step 25: Commit docs**

```bash
git add README.md CONTRIBUTING.md
git commit -m "docs: add VideoRecorderCallback to README and CONTRIBUTING"
```

---

## Dependency Graph

```mermaid
graph LR
    A["Card A: preprocess_obs()"] --> B["Card B: VideoRecorderCallback"]
```

Card A has no dependencies and can start immediately. Card B depends on Card A because `_run_eval_rollouts` calls `env.preprocess_obs()`. Within Card B, tasks B1–B6 are sequential (each builds on the previous).

## Trello Wave Execution

**Wave 1:** Card A only (no blockers).
**Wave 2:** Card B (after Card A merges to main).

Both cards have "Definition of Done" checklists already created on Trello. Map commits to checklist items:

| Card A Commit | Checklist Item |
|---|---|
| Task A2, Step 8 commit | Extract preprocess_obs, replace inline transforms, add tests, existing tests pass |

| Card B Commit | Checklist Item |
|---|---|
| Task B2, Step 5 | Implement VideoRecorderCallback class satisfying the Callback protocol |
| Task B3, Step 10 | Wrap env with RecordVideo in on_train_start |
| Task B3, Step 10 | Gracefully skip if env does not support render_mode='rgb_array' |
| Task B4, Step 15 | Save videos to run_dir/videos/ |
| Task B5, Step 20 | Support configurable recording triggers |
| Task B6, Step 21 | Add tests for callback lifecycle and protocol conformance |
| Task B6, Step 25 | Update docs with usage example |
