# VideoRecorderCallback Design

## Summary

Add a `VideoRecorderCallback` to `rltrain.callbacks` that records evaluation videos of agent behaviour during training. Uses gymnasium's `RecordVideo` wrapper on a separate persistent evaluation environment, triggered by default at each checkpoint.

## Motivation

Reward curves show *that* an agent is improving; videos show *how*. For rltrain's pedagogical goals, watching an agent go from random to competent is one of the most powerful learning tools available. Any gymnasium environment that supports `render_mode="rgb_array"` gets video recording for free via `RecordVideo` — this callback makes that accessible within rltrain's training loop.

## Design

### Constructor

```python
class VideoRecorderCallback:
    def __init__(
        self,
        *,
        env_fn: Callable[[], gym.Env] | None = None,
        num_episodes: int = 3,
        eval_trigger: Callable[[int], bool] | None = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        fps: int = 30,
    ) -> None:
```

**Parameters:**

- `env_fn` — Zero-arg callable returning a `gym.Env` with `render_mode="rgb_array"`. If `None`, auto-detects from `MDP.env.envs[0].spec.id` in `on_train_start`. The auto-detection creates a bare env without user-applied wrappers; pass `env_fn` explicitly when wrappers matter for the recording.
- `num_episodes` — Number of evaluation episodes to record at each trigger point. Default: 3.
- `eval_trigger` — Optional callable `(training_episode: int) -> bool` controlling when eval rollouts happen during training. When set, rollouts trigger in `on_episode_end` instead of the default `on_checkpoint`. For example, `lambda ep: ep % 50 == 0` records every 50th training episode.
- `video_length` — Fixed video length in frames. 0 means record full episodes. Forwarded to `RecordVideo`.
- `name_prefix` — Filename prefix for recorded videos. Default: `"rl-video"`.
- `fps` — Frames per second for the output video. Default: 30.

### Protocol Conformance

Satisfies the `Callback` protocol via structural subtyping — implements all five hook methods, no inheritance required.

### Lifecycle

| Hook | Behaviour |
|------|-----------|
| `on_train_start(agent, env, run_dir)` | Build eval env from `env_fn` or auto-detect from `env.env.envs[0].spec.id`. Wrap with `RecordVideo(eval_env, video_folder=run_dir / "videos", ...)`. Create the videos directory. |
| `on_step(agent, env, step)` | No-op. |
| `on_episode_end(agent, env, episode)` | If `eval_trigger` is set and `eval_trigger(episode)` returns True, run eval rollouts. Otherwise no-op. |
| `on_checkpoint(agent, env, run_dir)` | Default trigger: run `num_episodes` eval rollouts on the recording env. |
| `on_train_end(agent, env, run_dir)` | Run final recording, then close the eval env. |

### Eval Rollout

At each trigger point the callback runs `num_episodes` episodes on the eval env:

1. Reset the eval env.
2. Loop: call `agent.act(obs)` to get a distribution, sample an action, step the eval env.
3. Repeat until `terminated or truncated`.
4. `RecordVideo` handles frame capture and MP4 file writing transparently.

Actions are sampled stochastically from the policy distribution — this shows what the agent *actually does*, not an idealised mode. Deterministic eval can be added later if needed.

**Observation preprocessing:** The training `MDP` may apply observation preprocessing (e.g. channel-swap for image envs). The eval rollout must apply the same transformation so that `agent.act()` receives observations in the expected shape. The callback calls `env.preprocess_obs(obs)` on raw eval env observations before passing them to the agent. This depends on the `MDP.preprocess_obs()` encapsulation (see [Trello card #21](https://trello.com/c/ERxUf9Aa)).

### Auto-Detection Fallback

```python
def _make_env_from_mdp(self, env: MDP) -> gym.Env:
    spec = env.env.envs[0].spec
    if spec is None:
        raise RuntimeError(
            "Cannot auto-detect env — provide env_fn to VideoRecorderCallback"
        )
    return gym.make(spec.id, render_mode="rgb_array")
```

Creates a bare gymnasium env from the training env's registered spec. Does not replicate user-applied wrappers — the docstring will make this explicit.

### Output

Videos are saved to `run_dir/videos/`. This sits alongside the existing output structure:

```
run_dir/
    config/
    models/
    metrics.csv
    per_episode.svg
    per_sample.svg
    videos/               # NEW
        rl-video-episode-0.mp4
        rl-video-episode-1.mp4
        ...
```

Gymnasium's `RecordVideo` handles file naming automatically using the `name_prefix` and episode/step counters.

### Dependencies

- `gymnasium.wrappers.RecordVideo` — already available via the gymnasium dependency.
- No new external dependencies. Video encoding uses gymnasium's built-in MoviePy/ffmpeg integration.
- `MDP.preprocess_obs()` — [Trello card #21](https://trello.com/c/ERxUf9Aa). Should land before or alongside this callback.

### Graceful Degradation

If the eval env does not support `render_mode="rgb_array"` (either via auto-detection or a user-provided factory), the callback logs a warning and disables itself for the remainder of training. No error, no crash — just no videos.

## Testing

- **Protocol conformance** — `isinstance(VideoRecorderCallback(...), Callback)` passes. Follows the pattern in `tests/callbacks/test_protocol.py`.
- **Lifecycle** — Verify hooks are called in correct order and eval env is created/closed properly. Mock the eval env to avoid actual rendering in tests.
- **Auto-detection** — Test that the fallback correctly extracts `spec.id` from a real `MDP` wrapping a `SyncVectorEnv`.
- **Graceful skip** — Test that a non-renderable env triggers a warning and disables recording.
- **Video output** — Integration test with a simple env (e.g. CartPole with `render_mode="rgb_array"`) verifying that MP4 files are written to `run_dir/videos/`.

## Documentation

- Update `README.md` callbacks section with `VideoRecorderCallback` usage example.
- Add docstrings following the existing NumPy-style convention with backtick-wrapped parameter names.

## Usage Examples

**Minimal (auto-detect):**

```python
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
```

**Explicit factory with wrappers:**

```python
def make_recording_env():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = gym.wrappers.NormalizeObservation(env)
    return env

trainer = Trainer(
    agent, env,
    num_steps=100_000,
    checkpoint_steps=2500,
    run_dir=run_dir,
    callbacks=[
        CSVLoggerCallback(),
        PlotCallback(num_steps=100_000),
        CheckpointCallback(),
        VideoRecorderCallback(env_fn=make_recording_env, num_episodes=5),
    ],
)
```
