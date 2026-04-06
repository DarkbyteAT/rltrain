# Gradient Transforms

Compatibility shim that re-exports gradient transform classes from [samgria](https://github.com/DarkbyteAT/samgria). The implementations (SAM, ASAM, LAMPRollback, GradientTransform protocol) live in the samgria package; this module provides `rltrain.transforms.*` imports for backward compatibility.

## Available Re-exports

| Symbol | Source |
|--------|--------|
| `GradientTransform` | `samgria.GradientTransform` |
| `SAM` | `samgria.SAM` |
| `ASAM` | `samgria.ASAM` |
| `LAMPRollback` | `samgria.LAMPRollback` |

## Pipeline Execution

`Agent.learn()` runs the pipeline in this order:

1. `loss(*batch).backward()` -- populate gradients.
2. For each transform: `transform.apply(model, loss_fn, batch)` -- pre-descent.
3. `descend()` -- optimizer step.
4. For each transform: `transform.post_step(model)` -- post-descent.

Transforms compose: SAM modifies gradients before descent, then LAMP perturbs parameters after descent. The pipeline is configured via the `grad_transforms` key in agent JSON:

```json
"grad_transforms": [
    {"fqn": "samgria.SAM", "rho": 1e-2},
    {"fqn": "samgria.LAMPRollback", "eps": 5e-3, "rollback_len": 10}
]
```

Omitting the key (or passing an empty list) gives vanilla gradient descent.

## Adding Custom Transforms

Custom transforms can live anywhere importable and be referenced by FQN in the config. Implement a class with `apply(model, loss_fn, batch)` and `post_step(model)` methods matching the `GradientTransform` protocol -- structural subtyping handles conformance.
