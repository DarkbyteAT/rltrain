import torch.nn as nn


def cnn(channels: list[int], kernels: list, strides: list, act_fn: nn.Module | None = None):
    """Creates a feedforward CNN. Each layer is initialised with orthogonal weights, and uses a SiLU
    ctivation before the next convolution. The output of the CNN is output as a flattened batch of
    vectors. One may also specify an activation function to be used between layers; by default the
    network uses a ``SiLU`` (Swish) activation function.

    Parameters
    ----------
    ``channels`` : ``list[int]``
        The number of channels for each layer of the CNN.
    ``kernels`` : ``list[int | list[int]]``
        The size of the kernels between each layer of the CNN.
    ``strides`` : ``list[int | list[int]]``
        The stride of the filters between each layer of the CNN.

    Returns
    -------
    ``Sequential``
        A feed-forward CNN with the specified number of layers, flattening to a batch of feature
        vectors.
    """

    if act_fn is None:
        act_fn = nn.SiLU()

    layers = []

    for i in range(len(channels) - 1):
        layer = nn.Conv2d(channels[i], channels[i + 1], kernels[i], strides[i])
        nn.init.orthogonal_(layer.weight)
        layers.append(layer)
        layers.append(act_fn)

    layers.append(nn.Flatten())
    return nn.Sequential(*layers)
