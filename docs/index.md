# RLTrain

A modular PyTorch framework for deep reinforcement learning research.

JSON-driven configuration, composable neural network architectures, and robust optimisation techniques -- designed for comparing RL algorithms across environments with minimal boilerplate.

## Features

- **Algorithm hierarchy** -- Policy gradient, actor-critic, and Q-learning agents arranged in a clean inheritance chain where each level adds one concept.
- **JSON + FQN configuration** -- Agents, networks, optimisers, and environment wrappers are specified as fully-qualified class names resolved at runtime.
- **Composable networks** -- MLP, CNN, D2RL skip connections, and Random Fourier Features modules snap together sequentially.
- **Robust optimisation** -- SAM and LAMP for sharpness-aware training, toggleable per agent.
- **Callback protocol** -- Extensible training hooks for checkpointing, logging, plotting, and video recording.

## Quick links

| | |
|---|---|
| [Getting Started](getting-started.md) | Installation, first training run, understanding the output |
| [Algorithms](algorithms.md) | Agent hierarchy walkthrough with links to papers |
| [Configuration](configuration.md) | JSON config anatomy, FQN system, network modules |
| [API Reference](reference/index.md) | Auto-generated from source docstrings |
