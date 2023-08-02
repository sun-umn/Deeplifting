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
        self.scale = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):  # noqa
        return torch.sin(x * self.scale)


class AddOffset(nn.Module):
    """
    Class that adds the weights / bias offsets & is
    trainable for the structural optimization code
    """

    def __init__(self, scale=10):  # noqa
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(scale, dtype=torch.double), requires_grad=True
        )
        self.bias = nn.Parameter(
            torch.zeros(1),
            requires_grad=True,
        )

    def forward(self, x):  # noqa
        return x + (self.scale * self.bias)


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


# class DeepliftingSkipMLP(nn.Module):
#     """
#     Class that implements a standard MLP from
#     pytorch. We will utilize this as one of many
#     NN architectures for our deep lifting project.
#     Utiilizes skip connections.
#     """

#     def __init__(self, input_size, output_size, n):  # noqa
#         super(DeepliftingSkipMLP, self).__init__()

#         # Input layer 1 + BN
#         self.linear_layer1 = nn.Linear(input_size, n)
#         self.bn1 = nn.BatchNorm1d(n)

#         # Input layer 2 + BN
#         self.linear_layer2 = nn.Linear(n, n)
#         self.bn2 = nn.BatchNorm1d(n)

#         # Input layer 3 + BN
#         self.linear_layer3 = nn.Linear(n, n)
#         self.bn3 = nn.BatchNorm1d(n)

#         # Input layer 4 + BN
#         self.linear_layer4 = nn.Linear(n * 3, n)
#         self.bn4 = nn.BatchNorm1d(n)

#         # Output layer
#         self.output_float_layer = nn.Linear(n, output_size)
#         self.output_trunc_layer = nn.Linear(n, output_size)

#         # Activation
#         # self.activation = SinActivation()
#         self.activation = nn.LeakyReLU()

#         # Dropout
#         self.dropout = nn.Dropout(p=0.01)

#         # One of the things that we did with the topology
#         # optimization is also let the input be variable. Some
#         # of the problems we have looked at so far also are
#         # between bounds
#         self.x = nn.Parameter(torch.randn(input_size, input_size))

#     def forward(self, inputs=None):  # noqa
#         # First layer
#         output1 = self.linear_layer1(self.x)
#         output1 = self.bn1(output1)
#         # output1 = self.dropout(output1)
#         output1 = self.activation(output1)
#         output1 = AddOffset()(output1)

#         # Second layer
#         output2 = self.linear_layer2(output1)
#         output2 = self.bn2(output2)
#         # output2 = self.dropout(output2)
#         output2 = self.activation(output2)
#         output2 = AddOffset()(output2)

#         # Thrid layer
#         output3 = self.linear_layer3(output2)
#         output3 = self.bn3(output3)
#         # output3 = self.dropout(output3)
#         output3 = self.activation(output3)
#         output3 = AddOffset()(output3)

#         # Final layer
#         output = torch.cat(
#             (
#                 output1,
#                 output2,
#                 output3,
#             ),
#             axis=1,
#         )

#         # Get an output that allows float values
#         output = self.linear_layer4(output)
#         output = self.bn4(output)
#         output = self.activation(output)

#         # Final output
#         output_float = self.output_float_layer(output)
#         output_float = AddOffset()(output_float)

#         # Set up a region that focuses on integer values
#         # output_trunc = self.output_trunc_layer(output)
#         # output_trunc = nn.ReLU()(output_trunc)

#         return output_float, None  # , output_trunc


class DeepliftingBlock(nn.Module):
    def __init__(self, input_size, output_size, activation='sine'):
        super(DeepliftingBlock, self).__init__()
        # Define the activation
        self.activation = activation

        # Define the Linear layer
        self.linear = nn.Linear(input_size, output_size)

        # Define the Batch Normalization layer
        self.batch_norm = nn.BatchNorm1d(output_size)

    def forward(self, x):
        # Linear layer
        x = self.linear(x)

        # Batch Normalization
        x = self.batch_norm(x)

        # Sine activation function
        if self.activation == 'sine':
            x = torch.sin(x)
        elif self.activation == 'relu':
            x = nn.ReLU()(x)
        elif self.activation == 'LeakyReLU':
            x = nn.LeakyReLU()(x)

        return x


# Automating skip connection block
class DeepliftingSkipMLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        skip_every_n=1,
        activation='sine',
        agg_function='sum',
    ):
        super(DeepliftingSkipMLP, self).__init__()

        self.layers = nn.ModuleList()
        self.skip_every_n = skip_every_n
        self.agg_function = agg_function

        # Input layer
        self.layers.append(
            DeepliftingBlock(input_size, hidden_sizes[0], activation=activation)
        )

        # Hidden layers with skip connections
        for i in range(1, len(hidden_sizes)):
            self.layers.append(
                DeepliftingBlock(
                    hidden_sizes[i - 1], hidden_sizes[i], activation=activation
                )
            )
            if i % skip_every_n == 0:
                self.layers.append(
                    DeepliftingBlock(
                        hidden_sizes[i - skip_every_n],
                        hidden_sizes[i],
                        activation=activation,
                    )
                )

        # Output layer
        self.output_layer = DeepliftingBlock(
            hidden_sizes[-1], output_size, activation=activation
        )

        # One of the things that we did with the topology
        # optimization is also let the input be variable. Some
        # of the problems we have looked at so far also are
        # between bounds
        self.x = nn.Parameter(torch.randn(input_size, input_size))

    def forward(self, inputs=None):
        intermediate_connections = []
        x = self.x
        for i, layer in enumerate(self.layers):
            x_new = layer(x)
            if (i + 1) % self.skip_every_n == 0 and i != 0:
                intermediate_connections.append(x_new)
                x = x_new
            else:
                x = x_new

        # Stack the skipped connections and then sum
        # We will also make this configurable
        x = torch.stack(intermediate_connections)
        if self.agg_function == 'sum':
            x = torch.sum(x, axis=0)
        elif self.agg_function == 'average':
            x = torch.mean(x, axis=0)
        elif self.agg_function == 'max':
            x = torch.amax(x, axis=0)

        out = self.output_layer(x)
        return out, None
