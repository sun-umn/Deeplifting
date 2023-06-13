# third party
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Class that implements a standard MLP from
    pytorch. We will utilize this as one of many
    NN architectures for our deep lifting project.
    """

    def __init__(self, input_size, layer_sizes, output_size):  # noqa
        super(MLP, self).__init__()

        layers = []
        prev_layer_size = input_size
        for size in layer_sizes:
            linear_layer = nn.Linear(prev_layer_size, size)
            nn.init.orthogonal_(linear_layer.weight)  # Apply orthogonal initialization
            layers.append(linear_layer)
            layers.append(nn.BatchNorm1d(size))  # Add batch normalization
            layers.append(torch.sin)
            prev_layer_size = size

        # Output layer
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_layer_size, output_size)
        nn.init.orthogonal_(self.output_layer.weight)  # Apply orthogonal initialization

    def forward(self, x):  # noqa
        x = self.layers(x)
        x = self.output_layer(x)
        return x
