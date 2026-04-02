import torch as T
import torch.nn as nn


class SkipMLP(nn.Module):
    """Implements a MLP network with skip-connections between the input layer and all further
    layers. One may also specify an activation function to be used between layers; by default the
    network uses a ``SiLU`` (Swish) activation function."""

    def __init__(self, inputs: int, hiddens: list[int], outputs: int, act_fn: nn.Module | None = None):
        super().__init__()
        self.act_fn = act_fn if act_fn is not None else nn.SiLU()
        self.in_layer = nn.Linear(inputs, hiddens[0])
        self.hidden_layers = nn.ModuleList()
        self.out_layer = nn.Linear(inputs + hiddens[-1], outputs)

        last_hidden = inputs + hiddens[0]

        for next_hidden in hiddens:
            self.hidden_layers.append(nn.Linear(last_hidden, next_hidden))
            last_hidden = inputs + next_hidden

        nn.init.orthogonal_(self.in_layer.weight.data)
        for layer in self.hidden_layers:
            nn.init.orthogonal_(layer.weight.data)
        nn.init.orthogonal_(self.out_layer.weight.data)

    def forward(self, x: T.Tensor) -> T.Tensor:
        # Forward pass, then apply activations, concatenate the input, and repeat
        y = self.act_fn(self.in_layer(x))
        y = T.cat([x, y], dim=1)

        for layer in self.hidden_layers:
            y = self.act_fn(layer(y))
            y = T.cat([x, y], dim=1)

        return self.out_layer(y)
