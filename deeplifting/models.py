# third party
import torch
import torch.nn as nn


class SinActivation(nn.Module):
    """
    Class that makes the sin function
    an activation function
    """

    def __init__(self):  # noqa
        super(SinActivation, self).__init__()

    def forward(self, x):  # noqa
        return torch.sin(x)


class DeepliftingMLP(nn.Module):
    """
    Class that implements a standard MLP from
    pytorch. We will utilize this as one of many
    NN architectures for our deep lifting project.
    """

    def __init__(self, input_size, layer_sizes, output_size):  # noqa
        super(DeepliftingMLP, self).__init__()

        layers = []
        prev_layer_size = input_size
        for size in layer_sizes:
            linear_layer = nn.Linear(prev_layer_size, size)
            layers.append(linear_layer)
            layers.append(nn.BatchNorm1d(size))  # Add batch normalization
            layers.append(SinActivation())
            prev_layer_size = size

        # Output layer
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_layer_size, output_size)

        # One of the things that we did with the topology
        # optimization is also let the input be variable. Some
        # of the problems we have looked at so far also are
        # between bounds
        self.x = nn.Parameter(torch.randn(input_size, input_size))

    def forward(self, inputs=None):  # noqa
        output = self.layers(self.x)
        output = self.output_layer(output)
        return output


class DeepliftingSkipMLP(nn.Module):
    """
    Class that implements a standard MLP from
    pytorch. We will utilize this as one of many
    NN architectures for our deep lifting project.
    Utiilizes skip connections.
    """

    def __init__(self, input_size, output_size, n):  # noqa
        super(DeepliftingSkipMLP, self).__init__()

        # Input layer 1 + BN
        self.linear_layer1 = nn.Linear(input_size, n)
        self.bn1 = nn.BatchNorm1d(n)

        # Input layer 2 + BN
        self.linear_layer2 = nn.Linear(n, n)
        self.bn2 = nn.BatchNorm1d(n)

        # Input layer 3 + BN
        self.linear_layer3 = nn.Linear(n, n)
        self.bn3 = nn.BatchNorm1d(n)

        # Input layer 4 + BN
        self.linear_layer4 = nn.Linear(n * 3, n)
        self.bn4 = nn.BatchNorm1d(n)

        # Output layer
        self.output_float_layer = nn.Linear(n, output_size)
        self.output_trunc_layer = nn.Linear(n, output_size)

        # Activation
        # self.activation = SinActivation()
        self.activation = nn.LeakyReLU()

        # Dropout
        self.dropout = nn.Dropout(p=0.01)

        # One of the things that we did with the topology
        # optimization is also let the input be variable. Some
        # of the problems we have looked at so far also are
        # between bounds
        self.x = nn.Parameter(torch.randn(input_size, input_size))

    def forward(self, inputs=None):  # noqa
        # First layer
        output1 = self.linear_layer1(self.x)
        output1 = self.bn1(output1)
        # output1 = self.dropout(output1)
        output1 = self.activation(output1)

        # Second layer
        output2 = self.linear_layer2(output1)
        output2 = self.bn2(output2)
        # output2 = self.dropout(output2)
        output2 = self.activation(output2)

        # Thrid layer
        output3 = self.linear_layer3(output2)
        output3 = self.bn3(output3)
        # output3 = self.dropout(output3)
        output3 = self.activation(output3)

        # Final layer
        output = torch.cat(
            (
                output1,
                output2,
                output3,
            ),
            axis=1,
        )

        # Get an output that allows float values
        output = self.linear_layer4(output)
        output = self.bn4(output)
        output = self.activation(output)

        # Final output
        output_float = self.output_float_layer(output)

        # Set up a region that focuses on integer values
        output_trunc = self.output_trunc_layer(output)
        output_trunc = nn.LeakyReLU()(output_trunc)

        return output_float, output_trunc
