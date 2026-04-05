# Gradient Transforms

Composable optimisation techniques applied between `loss.backward()` and `descend()` inside `Agent.learn()`. Transforms modify gradients or parameters without touching the agent's core logic.

## GradientTransform Protocol

`GradientTransform` is a `@runtime_checkable` Protocol defined in `protocol.py` with two hooks:

| Hook | Phase | Signature | Purpose |
|------|-------|-----------|---------|
| `apply()` | Pre-descent | `(model, loss_fn, batch)` | Modify gradients or temporarily perturb parameters before the optimizer step |
| `post_step()` | Post-descent | `(model,)` | Operate on updated parameters after the optimizer step |

Both methods must be implemented. Use an empty body (`...`) for phases the transform does not participate in.

## Pipeline Execution

`Agent.learn()` runs the pipeline in this order:

1. `loss(*batch).backward()` -- populate gradients.
2. For each transform: `transform.apply(model, loss_fn, batch)` -- pre-descent.
3. `descend()` -- optimizer step.
4. For each transform: `transform.post_step(model)` -- post-descent.

Transforms compose: SAM modifies gradients before descent, then LAMP perturbs parameters after descent. The pipeline is configured via the `grad_transforms` key in agent JSON:

```json
"grad_transforms": [
    {"fqn": "rltrain.transforms.SAM", "rho": 1e-2},
    {"fqn": "rltrain.transforms.LAMPRollback", "eps": 5e-3, "rollback_len": 10}
]
```

Omitting the key (or passing an empty list) gives vanilla gradient descent.

## Built-in Transforms

| Transform | File | Phase | Description |
|-----------|------|-------|-------------|
| `SAM` | `sam.py` | Pre-descent | Sharpness-Aware Minimisation. Perturbs parameters in the normalised gradient direction by `rho`, recomputes loss and gradient at the perturbed point, then restores original parameters with the new gradient. Encourages flat minima. |
| `ASAM` | `asam.py` | Pre-descent | Adaptive SAM. Like SAM, but the perturbation direction is scaled by squared parameter magnitude (`|theta|^2 * grad`), making sharpness invariant to parameter rescaling. |
| `LAMPRollback` | `lamp.py` | Post-descent | Local-Averaging over Multiple Perturbations. After each descent step, injects uniform noise scaled by parameter magnitude and accumulates a moving average. After `rollback_len` steps, rolls parameters back to the average. Designed to compose after SAM/ASAM. |

All transforms use `parameters_to_vector` / `vector_to_parameters` for efficient parameter manipulation, and `get_grad` / `set_grad` from `rltrain.utils` for gradient vectorisation.

## How to Add a New Transform

1. Create a new file in `rltrain/transforms/`.
2. Implement a class with `apply(model, loss_fn, batch)` and `post_step(model)` methods matching the `GradientTransform` protocol.
3. Re-export from `rltrain/transforms/__init__.py`.
4. The transform is now usable in any agent JSON config via its FQN.

The transform does not need to inherit from anything -- structural subtyping handles protocol conformance. It can also live outside this package (anywhere importable) and be referenced by FQN in the config.
