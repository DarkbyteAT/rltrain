# Utils

Shared utilities used across the framework: the FQN builder system, device resolution, and mathematical helpers for gradient manipulation and discounting.

## FQN Builder System (`builders/`)

The builder subpackage is the backbone of rltrain's JSON-driven configuration. It dynamically imports classes and constructs object graphs from JSON configs at runtime.

### Core Functions

| Function | File | Purpose |
|----------|------|---------|
| `load(fqn)` | `load.py` | Import a module, class, or function by fully-qualified name string |
| `resolve(cfg)` | `load.py` | Recursively resolve a JSON config tree -- dicts with `"fqn"` keys become constructed objects, `"deferred": true` wraps in `functools.partial` |
| `agent(fqn, model, opt, device, **kwargs)` | `agent.py` | Build an `Agent` from JSON config: resolves network modules into `nn.ModuleDict`, wraps optimizers as deferred partials, resolves gradient transforms |
| `env(id, wrappers, **kwargs)` | `env.py` | Build a `SyncVectorEnv` with gymnasium wrappers applied in order |
| `eval_env(id, wrappers, **kwargs)` | `env.py` | Build a single wrapped env with `render_mode="rgb_array"` for evaluation |
| `load_agent(run_dir, checkpoint, device)` | `checkpoint.py` | Reconstruct a trained agent from a saved run directory (config + state dict) |

### FQN Conventions

FQNs must resolve through `__init__.py` re-exports. Always use the shortest public name:

- `toblox.SkipMLP` (correct) -- resolves via `toblox/__init__.py`
- `toblox.d2rl.SkipMLP` (avoid) -- unnecessarily deep path

## Device Resolution (`device.py`)

`resolve_device(device_str)` maps user-facing device strings to concrete `torch.device` instances. The `"auto"` mode probes backends in priority order: CUDA, then MPS, then CPU. Raises `ValueError` if a specific backend is requested but unavailable.

## Math Helpers

| Function | File | Purpose |
|----------|------|---------|
| `discount(xs, dones, factor)` | `discount.py` | Compute discounted cumulative sums (returns, GAE) with episode boundary handling via the `dones` mask |
| `center(x)` | `center.py` | Standardise a tensor to zero mean and unit variance (whitening) |
| `lerp(input, target, step)` | `lerp.py` | Linear interpolation between two values (used for target network soft updates) |
| `get_grad(params)` | re-exported from `samgria` | Flatten all parameter gradients into a single vector |
| `set_grad(grads, params)` | re-exported from `samgria` | Scatter a gradient vector back into parameter `.grad` attributes |

The gradient utilities (`get_grad`, `set_grad`) are re-exported from [samgria](https://github.com/DarkbyteAT/samgria) and used by SAM, ASAM, and other transforms that need to read, manipulate, and restore gradients as flat vectors.

## How to Add a New Utility

1. Create a new file in `rltrain/utils/` (or `rltrain/utils/builders/` for builder functions).
2. Re-export from the appropriate `__init__.py`.
3. Keep utilities pure and stateless where possible -- they are shared infrastructure, not module-specific logic.
