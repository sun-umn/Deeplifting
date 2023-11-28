# stdlib
import random

# third party
import numpy as np
import torch
import torch.nn as nn

# class SinActivation(nn.Module):
#     """
#     Class that makes the sin function
#     an activation function
#     """

#     def __init__(self):  # noqa
#         super(SinActivation, self).__init__()
#         # self.amplitude = nn.Parameter(torch.pi * torch.ones(1), requires_grad=True)
#         self.scale = nn.Parameter(torch.ones(1), requires_grad=True)
#         self.shift = nn.Parameter(torch.zeros(1), requires_grad=True)
#         self.y_shift = nn.Parameter(torch.zeros(1), requires_grad=True)

#     def forward(self, x):  # noqa
#         return torch.sin((x - self.shift) * self.scale)


# class AddOffset(nn.Module):
#     """
#     Class that adds the weights / bias offsets & is
#     trainable for the structural optimization code
#     """

#     def __init__(self, scale=10):  # noqa
#         super().__init__()
#         self.scale = nn.Parameter(
#             torch.tensor(scale, dtype=torch.double), requires_grad=True
#         )
#         self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)

#     def forward(self, x):  # noqa
#         return x + (self.scale * self.bias)


# class DeepliftingMLP(nn.Module):
#     """
#     Class that implements a standard MLP from
#     pytorch. We will utilize this as one of many
#     NN architectures for our deep lifting project.
#     """

#     def __init__(self, input_size, layer_sizes, output_size):  # noqa
#         super(DeepliftingMLP, self).__init__()

#         layers = []
#         prev_layer_size = input_size
#         for size in layer_sizes:
#             linear_layer = nn.Linear(prev_layer_size, size)
#             layers.append(linear_layer)
#             layers.append(nn.BatchNorm1d(size))  # Add batch normalization
#             layers.append(SinActivation())
#             prev_layer_size = size

#         # Output layer
#         self.layers = nn.Sequential(*layers)
#         self.output_layer = nn.Linear(prev_layer_size, output_size)

#         # One of the things that we did with the topology
#         # optimization is also let the input be variable. Some
#         # of the problems we have looked at so far also are
#         # between bounds
#         # self.x = nn.Parameter(torch.randn(input_size, input_size))

#     def forward(self, inputs=None):  # noqa
#         output = self.layers(self.x)
#         output = self.output_layer(output)
#         return output


# class DeepliftingBlock(nn.Module):
#     def __init__(self, input_size, output_size, activation='sine', include_bn=False):
#         super(DeepliftingBlock, self).__init__()
#         # Define the activation
#         self.activation = activation
#         self.include_bn = include_bn

#         if self.activation == 'sine':
#             self.activation_layer = SinActivation()
#         elif self.activation == 'relu':
#             self.activation_layer = nn.ReLU()
#         elif self.activation == 'leaky_relu':
#             self.activation_layer = nn.LeakyReLU()
#         else:
#             self.activation_layer = nn.Identity()

#         # Define the Linear layer
#         self.linear = nn.Linear(input_size, output_size)

#         # Define a initlization scheme
#         if self.activation == 'relu':
#             # initialize the weights
#             nn.init.kaiming_normal_(
#                 self.linear.weight,
#                 mode='fan_in',
#                 nonlinearity=self.activation,
#             )
#             # Initalize the bias to zero
#             nn.init.zeros_(self.linear.bias)

#         elif self.activation == 'leaky_relu':
#             # Initialize the weights for leaky relu
#             nn.init.kaiming_normal_(
#                 self.linear.weight,
#                 mode='fan_in',
#                 nonlinearity=self.activation,
#             )
#             # Initailize the bias to zero
#             nn.init.zeros_(self.linear.bias)

#         elif self.activation == 'identity':
#             # Initialize the weights for leaky relu
#             nn.init.kaiming_normal_(
#                 self.linear.weight,
#                 mode='fan_in',
#                 nonlinearity='relu',
#             )
#             # Initailize the bias to zero
#             nn.init.zeros_(self.linear.bias)

#         # Define the Batch Normalization layer
#         # self.batch_norm = nn.BatchNorm1d(output_size)
#         self.batch_norm = nn.LayerNorm(output_size)

#     def forward(self, x):
#         # Linear layer
#         x = self.linear(x)

#         if self.include_bn:
#             # Batch Normalization
#             x = self.batch_norm(x)

#         if self.activation not in ('sine', 'relu', 'leaky_relu'):
#             return x

#         # Activation function
#         x = self.activation_layer(x)

#         return x


# class DeepliftingScalingBlock(nn.Module):
#     def __init__(self, bounds, output_activation, dimensions=2, scale=1):
#         super(DeepliftingScalingBlock, self).__init__()
#         # Define the bounds for the class
#         self.bounds = bounds
#         self.dimensions = dimensions
#         self.scale = scale

#         # Define the scaler based on the final activation layer
#         # of the output
#         self.output_activation = output_activation

#     def forward(self, outputs):
#         # Let's try out trick from topology
#         # optimization instead of relying on the
#         # inequality constraint
#         # If we map x and y to [0, 1] and then shift
#         # the interval we can accomplist the same
#         # thing we can use a + (b - a) * x

#         # For sin [-1, 1]
#         # c + (d - c) / (b - a) * (x - a)
#         # c + (d - c) / (2) * (x + 1)

#         # Try updating the way we define the bounds
#         if self.dimensions > 2:
#             # Try: Get the first bound and confine it this way
#             # want to see if this is a memory leak - this was definetly a part of it
#             a, b = self.bounds[0]
#             if self.output_activation != 'sine':
#                 return a + (b - a) / 2.0 * (torch.sin(outputs) + 1)
#             else:
#                 return a + (b - a) / 2.0 * (torch.sin(outputs) + 1)

#         else:
#             x_values_float = []
#             for index, cnstr in enumerate(self.bounds):
#                 a, b = cnstr
#                 if (a is None) and (b is None):
#                     x_constr = outputs[index]
#                 elif (a is None) or (b is None):
#                     x_constr = torch.clamp(outputs[index], min=a, max=b)

#                 # Being very explicit about this condition just in case
#                 # to avoid weird behavior
#                 elif (a is not None) and (b is not None):
#                     if self.output_activation != 'sine':
#                         x_constr = a + (b - a) / 2.0 * (outputs[:, index] + 1)
#                     else:
#                         x_constr = a + (b - a) / 2.0 * (outputs[:, index] + 1)
#                 x_values_float.append(x_constr)
#             x = torch.stack(x_values_float, axis=1)

#             # Clean
#             del x_values_float
#             torch.cuda.empty_cache()

#             return x


# # Automating skip connection block
# class DeepliftingSkipMLP(nn.Module):
#     def __init__(
#         self,
#         input_size,
#         hidden_sizes,
#         output_size,
#         bounds,
#         *,
#         skip_every_n=1,
#         activation='sine',
#         output_activation='sine',
#         agg_function='sum',
#         include_bn=True,
#         seed=0,
#     ):
#         super(DeepliftingSkipMLP, self).__init__()
#         # Set the seed
#         set_seed(seed)

#         self.layers = nn.ModuleList()
#         self.skip_every_n = skip_every_n
#         self.agg_function = agg_function
#         self.include_bn = include_bn
#         self.first_hidden_size = 5 * output_size  # hidden_sizes[0] // 2
#         self.scale = torch.randn(1) * 2 * torch.pi
#         self.bounds = bounds

#         # Input layer
#         self.layers.append(
#             DeepliftingBlock(
#                 self.first_hidden_size, hidden_sizes[0], activation=activation
#             )
#         )

#         # Hidden layers with skip connections
#         for i in range(1, len(hidden_sizes)):
#             self.layers.append(
#                 DeepliftingBlock(
#                     hidden_sizes[i - 1],
#                     hidden_sizes[i],
#                     activation=activation,
#                     include_bn=include_bn,
#                 )
#             )
#             if i % skip_every_n == 0:
#                 self.layers.append(
#                     DeepliftingBlock(
#                         hidden_sizes[i - skip_every_n],
#                         hidden_sizes[i],
#                         activation=activation,
#                         include_bn=include_bn,
#                     )
#                 )

#         # Output layer
#         self.output_layer = DeepliftingBlock(
#             hidden_sizes[-1],
#             output_size,
#             activation='identity',
#         )

#         self.output_layer_activation = SinActivation()

#         # # Linear scaling layer
#         # self.linear_scaling_layer = nn.Linear(output_size, output_size)

#         # Final scaling layer
#         self.scaling_layer = DeepliftingScalingBlock(
#             bounds=bounds,
#             output_activation='sine',
#             dimensions=output_size,
#             scale=self.scale,
#         )

#         # One of the things that we did with the topology
#         # optimization is also let the input be variable. Some
#         # of the problems we have looked at so far also are
#         # between bounds
#         # self.x = nn.Parameter(torch.randn(input_size, self.first_hidden_Size), requires_grad=False)  # noqa
#         # self.input_norm = nn.LayerNorm(self.first_hidden_size)

#     def forward(self, inputs=None):
#         intermediate_connections = []
#         x = inputs

#         for i, layer in enumerate(self.layers):
#             x_new = layer(x)
#             if (i + 1) % self.skip_every_n == 0 and i != 0:
#                 intermediate_connections.append(x_new)
#                 x = x_new
#             else:
#                 x = x_new

#         # Stack the skipped connections and then sum
#         # We will also make this configurable
#         if self.agg_function == 'identity':
#             # Final output layer
#             out = self.output_layer(x)
#         elif self.agg_function == 'sum':
#             x = torch.stack(intermediate_connections)
#             x = torch.sum(x, axis=0)
#             # Final output layer
#             out = self.output_layer(x)
#         elif self.agg_function == 'average':
#             x = torch.stack(intermediate_connections)
#             x = torch.mean(x, axis=0)
#             # Final output layer
#             out = self.output_layer(x)
#         elif self.agg_function == 'max':
#             x = torch.stack(intermediate_connections)
#             x = torch.amax(x, axis=0)
#             # Final output layer
#             out = self.output_layer(x)

#         # Output activation layer
#         out = self.output_layer_activation(out)

#         del intermediate_connections
#         torch.cuda.empty_cache()

#         # # # Run through the scaling layer
#         # # out = self.linear_scaling_layer(out)
#         if self.bounds is not None:
#             out = self.scaling_layer(out)
#         return out


# Relu Based Network #
def set_seed(seed):
    """
    Function to set the seed for the run
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class SinActivation(nn.Module):
    """
    Class that makes the sin function
    an activation function
    """

    def __init__(self):  # noqa
        super(SinActivation, self).__init__()
        # self.amplitude = nn.Parameter(torch.pi * torch.ones(1), requires_grad=True)
        self.scale = nn.Parameter(torch.ones(1), requires_grad=True)
        self.shift = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.y_shift = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):  # noqa
        return torch.sin((x - self.shift) * self.scale)


class DeepliftingScalingBlock(nn.Module):
    def __init__(self, bounds, dimensions=2, scale=1):
        super(DeepliftingScalingBlock, self).__init__()
        # Define the bounds for the class
        self.bounds = bounds
        self.dimensions = dimensions
        self.scale = scale

        # Define the scaler based on the final activation layer
        # of the output

    def forward(self, outputs):
        # Let's try out trick from topology
        # optimization instead of relying on the
        # inequality constraint
        # If we map x and y to [0, 1] and then shift
        # the interval we can accomplist the same
        # thing we can use a + (b - a) * x

        # For sin [-1, 1]
        # c + (d - c) / (b - a) * (x - a)
        # c + (d - c) / (2) * (x + 1)

        # We will need to change the format of our problems
        # but this should work well
        a = torch.tensor(self.bounds['lower_bounds'])
        b = torch.tensor(self.bounds['upper_bounds'])

        x = a + ((b - a) / 2.0 * (outputs + 1))

        return x


class ReluDeepliftingBlock(nn.Module):
    """
    Layer for deeplifting that relies on the ReLU activation layer
    and kaiming / he initialization
    """

    def __init__(
        self, input_size, output_size, include_weight_init=True, include_bn=False
    ):
        super(ReluDeepliftingBlock, self).__init__()
        # Define the activation
        self.include_bn = include_bn

        # ReLU activation layer
        self.activation_layer = nn.ReLU()

        # Define the Linear layer
        self.linear = nn.Linear(input_size, output_size)

        # Let's actually make this configurable so we can run
        # experiments with and without the weight initialization
        # Define a initlization scheme
        # initialize the weights
        # Also, we will only consider the ReLU activation layer
        if include_weight_init:
            nn.init.kaiming_uniform_(
                self.linear.weight,
                mode='fan_in',
                nonlinearity='relu',
            )

        # Initailize the bias to zero
        nn.init.zeros_(self.linear.bias)

        # Define the Batch Normalization layer
        # self.batch_norm = nn.BatchNorm1d(output_size)
        self.batch_norm = nn.LayerNorm(output_size)

    def forward(self, x):
        # Linear layer
        x = self.linear(x)

        if self.include_bn:
            # Batch Normalization
            x = self.batch_norm(x)

        # Activation function
        x = self.activation_layer(x)

        return x


class AlignmentLayer(nn.Module):
    """
    Class that makes the sin function
    an activation function
    """

    def __init__(self, output_size):  # noqa
        super(AlignmentLayer, self).__init__()
        # self.amplitude = nn.Parameter(torch.pi * torch.ones(1), requires_grad=True)
        self.alignment = nn.Parameter(torch.ones(output_size))

    def forward(self, x):  # noqa
        return x * self.alignment


# Build a neural network that does not have skip connections
# Automating skip connection block
class ReLUDeepliftingMLP(nn.Module):
    def __init__(
        self,
        initial_hidden_size,
        hidden_sizes,
        output_size,
        bounds,
        *,
        include_weight_initialization=True,
        include_bn=True,
        initial_layer_type='embedding',
        seed=0,
    ):
        super(ReLUDeepliftingMLP, self).__init__()
        # Set the seed
        set_seed(seed)

        self.layers = nn.ModuleList()
        self.include_bn = include_bn
        self.initial_hidden_size = initial_hidden_size
        self.bounds = bounds
        self.initial_layer_type = initial_layer_type
        self.include_weight_initialization = include_weight_initialization

        # The first input layer
        if self.initial_layer_type == 'embedding':
            self.input_layer = nn.Embedding(self.initial_hidden_size, hidden_sizes[0])

        elif self.initial_layer_type == 'linear':
            # Create different intialization for the first input layer
            self.input_layer = nn.Linear(self.initial_hidden_size, hidden_sizes[0])

        else:
            raise ValueError(f'{self.initial_layer_type} is not a valid option!')

        self.layers.append(self.input_layer)

        # Hidden layers with skip connections
        for i in range(1, len(hidden_sizes)):
            self.layers.append(
                ReluDeepliftingBlock(
                    hidden_sizes[i - 1],
                    hidden_sizes[i],
                    include_bn=include_bn,
                )
            )

        # Output layer
        # Each of the previous layers will have relu activation functions
        # in this layer we want to only use the identity function and then
        # we can pass it through a sine activation to map data -> [-1, 1]
        self.linear_output = nn.Linear(hidden_sizes[-1], output_size)

        # Initialize the weights for the input of the sine layer
        # initialize the weights
        if self.include_weight_initialization:
            nn.init.kaiming_uniform_(
                self.linear_output.weight,
                mode='fan_in',
                nonlinearity='relu',
            )

            # Initailize the bias to zero
            nn.init.zeros_(self.linear_output.bias)

        # Final scaling layer
        self.scaling_layer = DeepliftingScalingBlock(
            bounds=bounds,
            dimensions=output_size,
        )

        # Output activation layer
        self.output_activation_layer = SinActivation()

        # Initialization parameter
        self.alignment_layer = AlignmentLayer(output_size=output_size)

    def forward(self, inputs=None):
        x = inputs

        # Iterate over the layers to build the MLP
        for i, layer in enumerate(self.layers):
            x = layer(x)

        # Put it through the output layer
        x = self.linear_output(x)

        # If there is an embedding layer then swap axes and
        # collapse data
        if self.initial_layer_type == 'embedding':
            x = x.swapaxes(2, 1)
            x = x.mean(axis=-1)

        # Small layer to align the outputs of the neural network
        x = x.mean(axis=0)

        x = self.alignment_layer(x)
        out = self.output_activation_layer(x)

        if self.bounds is not None:
            out = self.scaling_layer(out)
        return out
