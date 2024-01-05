# stdlib
import random

# third party
import numpy as np
import torch
import torch.nn as nn


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


# Create a custom global normalization layer for pytorch
class GlobalNormalization(nn.Module):
    """
    Class that computes the global normalization that we
    saw in the structural optimization code
    """

    def __init__(self, epsilon=1e-6):  # noqa
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):  # noqa
        var, mean = torch.var_mean(x, unbiased=False)
        net = x
        net = net - mean
        net = net * torch.rsqrt(var + self.epsilon)
        return net


class SinActivation(nn.Module):
    """
    Class that makes the sin function
    an activation function
    """

    def __init__(self, include_amplitude=False):  # noqa
        super(SinActivation, self).__init__()
        self.include_amplitude = include_amplitude
        self.amplitude = nn.Parameter(torch.pi * torch.tensor(1), requires_grad=True)
        self.scale = nn.Parameter(torch.pi * torch.tensor(1), requires_grad=True)

    def forward(self, x):  # noqa
        if self.include_amplitude:
            return self.amplitude * torch.sin(x * self.scale)

        return torch.sin(x * self.scale)


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
        self.alignment = nn.Parameter(torch.randn(output_size))

    def forward(self, x):  # noqa
        return x - self.alignment


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

        # Initialization for initial layer
        nn.init.orthogonal_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)

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

        # Initial input
        x = self.layers[0](x)

        # Iterate over the layers to build the MLP
        for i, layer in enumerate(self.layers[1:]):
            # We need at least one output from the first hidden layer
            # before we can accumulate skip connections
            if i > 0:
                x_new = layer(x)
                x = x + x_new
            else:
                # Output for the initial layer
                x = layer(x)

        # Put it through the output layer
        x = self.linear_output(x)

        # If there is an embedding layer then swap axes and
        # collapse data
        if self.initial_layer_type == 'embedding':
            x = x.swapaxes(2, 1)
            x = x.mean(axis=-1)

        out = self.output_activation_layer(x)
        out = out.mean(axis=0)

        if self.bounds is not None:
            out = self.scaling_layer(out)

        return out


# Building a manual simple MLP for deeplifting to test with
# SGD
class DeepliftingSimpleMLP(nn.Module):
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
        super(DeepliftingSimpleMLP, self).__init__()

        # Set the seed
        set_seed(seed)

        # Setup the first linear layer
        self.linear1 = nn.Linear(initial_hidden_size, hidden_sizes)
        torch.nn.init.kaiming_uniform_(self.linear1.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.linear1.bias)

        # Set up a second linear layer
        self.linear2 = nn.Linear(hidden_sizes, hidden_sizes)
        torch.nn.init.kaiming_uniform_(self.linear2.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.linear2.bias)

        # Set up a third linear layer
        self.linear3 = nn.Linear(hidden_sizes, output_size)
        torch.nn.init.kaiming_uniform_(self.linear3.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.linear3.bias)

        # Final scaling layer
        self.scaling_layer = DeepliftingScalingBlock(
            bounds=bounds,
            dimensions=output_size,
        )

        self.sine_activation = SinActivation()

    def forward(self, inputs):
        # First input layer
        x = self.linear1(inputs)
        x = nn.ReLU()(x)
        x = GlobalNormalization()(x)

        # Second input layer
        x = self.linear2(x)
        x = nn.ReLU()(x)
        x = GlobalNormalization()(x)

        # Third input layer
        x = self.linear3(x)

        # Push through the sine layer and the
        # mapping
        x = self.sine_activation(x)
        x = self.scaling_layer(x)

        # Take the mean of the outputs
        x = x.mean(axis=0).flatten()

        return x
