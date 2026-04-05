# Neural Network Modules

Reusable network building blocks for agents. All modules use orthogonal weight initialisation and are designed to be composed via JSON configuration and the FQN builder.

## Modules

| Module | Type | File | FQN |
|--------|------|------|-----|
| `mlp` | Factory function | `mlp.py` | `rltrain.nn.mlp` |
| `cnn` | Factory function | `cnn.py` | `rltrain.nn.cnn` |
| `SkipMLP` | `nn.Module` class | `d2rl.py` | `rltrain.nn.SkipMLP` |
| `RFF` | `nn.Module` class | `rff.py` | `rltrain.nn.RFF` |

### `mlp(features, act_fn=nn.Tanh)`

Creates an `nn.Sequential` MLP from a tuple of layer sizes. Activation functions are placed between all layers except the last. Each `nn.Linear` layer gets orthogonal weight init.

### `cnn(channels, kernels, strides, act_fn=nn.SiLU)`

Creates an `nn.Sequential` CNN with `nn.Conv2d` layers followed by `nn.Flatten()`. Each conv layer gets orthogonal weight init. Default activation is SiLU (Swish).

### `SkipMLP(inputs, hiddens, outputs, act_fn=nn.SiLU)`

D2RL-style MLP where the raw input is concatenated with the output of every hidden layer, creating skip connections from input to all depths. This improves gradient flow for deeper networks. Default activation is SiLU.

### `RFF(in_features, out_features, bandwidth=10.0)`

Random Fourier Features projection layer. Subclasses `nn.Linear` with frozen random weights, projecting inputs through `sin` and `cos` to produce spectral feature encodings. `out_features` must be even (split equally between sin and cos components).

## Orthogonal Initialisation

Every module in this package initialises linear and conv layer weights with `nn.init.orthogonal_()`. This is a project-wide invariant -- all new modules must follow this convention.

## FQN Resolution and JSON Composition

Networks are specified in agent JSON configs as a mapping of names to lists of module specs under the `model` key. Each spec is a dict with an `"fqn"` key and constructor kwargs. The builder wraps each list in `nn.Sequential`, so multiple modules compose into a pipeline:

```json
{
    "model": {
        "actor": [{"fqn": "rltrain.nn.SkipMLP", "inputs": 4, "hiddens": [256, 256], "outputs": 2}],
        "critic": [{"fqn": "rltrain.nn.mlp", "features": [4, 256, 256, 1]}]
    }
}
```

For visual environments, an embedding CNN feeds into an MLP via the `embedding` key, which agents like `VanillaAC` compose with actor/critic heads using `nn.Sequential`.

## How to Add a New Module

1. Create a new file in `rltrain/nn/`.
2. Implement either a factory function returning `nn.Sequential` or an `nn.Module` subclass.
3. Apply `nn.init.orthogonal_()` to all linear and conv layer weights.
4. Re-export from `rltrain/nn/__init__.py` so the FQN resolves via the shortest public name (e.g. `rltrain.nn.MyModule`, not `rltrain.nn.my_file.MyModule`).
5. Verify the module works with the FQN builder by testing it in a JSON config.
