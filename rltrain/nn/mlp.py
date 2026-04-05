import torch.nn as nn


def mlp(features: tuple[int], act_fn: nn.Module = nn.Tanh) -> nn.Sequential:
    """Creates a multi-layer perceptron network with the given activation function between each
    layer. Each layer is initialised with orthogonal weights.

    Parameters
    ----------
    ``features`` : ``tuple[int]``
        The number of features for each layer of the MLP.
    ``act_fn`` : ``Module``
        The activation function to apply between each layer (excluding the final layer).

    Returns
    -------
    ``Sequential``
        A feed-forward network with the specified number of layers.
    """

    layers = []

    for i in range(len(features) - 1):
        layer = nn.Linear(features[i], features[i + 1])
        nn.init.orthogonal_(layer.weight)
        layers.append(layer)

        # Only adds activation function to intermediate layers
        if i < len(features) - 2:
            layers.append(act_fn())

    return nn.Sequential(*layers)
