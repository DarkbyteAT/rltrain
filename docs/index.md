# RLTrain

A modular JAX framework for deep reinforcement learning research.

JSON-driven configuration, Equinox-native pure-function agents, and three training-loop strategies (Python, `lax.scan`, `pmap`) — designed for comparing RL algorithms across environments with minimal boilerplate.

## Features

- **Algorithm coverage** — Policy gradient (VanillaPG → REINFORCE → VanillaAC → AdvantageAC → PPO, plus SPO), value-based (VanillaDQN, DoubleDQN, distributional C51), and SAC for both discrete and continuous action spaces.
- **JSON + FQN configuration** — Agents, networks, optimisers, and environments are specified as fully-qualified class names resolved at runtime. PRNG sub-keys are auto-spliced into constructors.
- **Loop strategies** — `Trainer` auto-selects between `PythonLoop`, `ScanLoop` (in-XLA rollouts via `lax.scan`), and `PmapLoop` (multi-device) based on the env's capability tuple.
- **Pure-functional agents** — Agents are `eqx.Module`s with no array leaves; all mutable state lives in a `chex.dataclass` threaded through `learn`. The whole pipeline composes with `jax.grad`, `jax.vmap`, and `jax.lax.scan`.
- **Callback protocol** — Five-hook `Protocol` for checkpointing, CSV logging, plotting, and video recording. Hooks fire Python-side at segment boundaries — no `io_callback` footguns.

## Quick links

| | |
|---|---|
| [Getting Started](getting-started.md) | Installation, first training run, understanding the output |
| [Algorithms](algorithms.md) | Agent inventory and design notes |
| [Configuration](configuration.md) | JSON config anatomy and the FQN builder |
| [Ecosystem](ecosystem.md) | How rltrain composes with samgria, xptrack, and toblox |
| [API Reference](reference/index.md) | Auto-generated from source docstrings |
