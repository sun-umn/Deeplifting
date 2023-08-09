# stdlib
import glob
import os
import random
from functools import partial

# third party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pygranso.pygransoStruct import pygransoStruct

# first party
from deeplifting.models import DeepliftingSkipMLP


def build_deeplifting_results(file_structure, epsilon=5.5e-4):
    """
    Function that can help us easily create results.
    file_structure is the naming convention of the file.
    We can use a wild-card and load in multiple files at
    once.
    """
    directory = os.getcwd()
    results_path = os.path.join(directory, file_structure)

    # Get all files
    files = glob.glob(results_path)

    # Load in the parquet files
    df = pd.read_parquet(files)

    # Create the hit variable
    df['hits'] = np.abs(df['global_minimum'] - df['f'])

    # Return the aggregated results
    return df.groupby('problem_name').agg({'hits': 'mean'})


def load_deeplifting_model(model_path, config):
    """
    Load a saved model from path. The input config
    will be the parameters of the neural network
    and will contain a path with the saved file
    """
    input_size = config['input_size']
    hidden_sizes = config['hidden_sizes']
    dimensions = config['dimensions']
    bounds = config['bounds']
    activation = config['activation']
    output_activation = config['output_activation']
    agg_function = config['agg_function']
    seed = config['seed']

    # Initialize the model
    model = DeepliftingSkipMLP(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=dimensions,
        bounds=bounds,
        skip_every_n=1,
        activation=activation,
        output_activation=output_activation,
        agg_function=agg_function,
        seed=seed,
    )

    # Load the model
    model.load_state_dict(torch.load(model_path))


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


def initialize_vector(size, bounds):
    if len(bounds) != size:
        raise ValueError("The number of bounds must match the size of the vector")

    vector = [np.random.uniform(low, high) for low, high in bounds]
    return np.array(vector)


def add_jitter(points, jitter_amount=0.05):
    jitter = np.random.uniform(-jitter_amount, jitter_amount, points.shape)
    return points + jitter


def create_contour_plot(problem_name, problem, models, trajectories, colormap='OrRd_r'):
    """
    Function that will build out the plots and the solution
    found for the optimization. For our purposes we will mainly
    be interested in the deep learning results.
    """
    # Get the objective function
    objective = problem['objective']

    # Get the bounds for the problem
    x_bounds, y_bounds = problem['bounds']

    # Separate into the minimum and maximum bounds
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    # Some of the problems that we are exploring do not
    # have bounds but we will want to plot them
    if x_max is None:
        x_max = 1e6
    if x_min is None:
        x_min = -1e6
    if y_max is None:
        y_max = 1e6
    if y_min is None:
        y_min = -1e6

    # Create a grid of points
    x = np.linspace(x_min, x_max, 1000)
    y = np.linspace(y_min, y_max, 1000)
    x, y = np.meshgrid(x, y)

    # For the function inputs we need results and a trial
    # Create dummy data
    results = np.zeros((1, 1, 3))
    trial = 0

    # Put objective function in a wrapper
    objective_f = partial(objective, results=results, trial=trial, version='numpy')

    # Create a grid of vectors
    grid = np.stack((x, y), axis=-1)

    # Apply the function on each point of the grid
    z = np.apply_along_axis(objective_f, 2, grid)

    # Create a figure
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)

    contour = ax1.contour(x, y, z, levels=10, cmap=colormap, alpha=0.5)
    fig.colorbar(contour, ax=ax1)

    # Define colors and markers for the models
    colors = ['black', 'black', 'black', 'black', 'black']
    markers = ['o', 's', '^', 'd', 'p']

    # Plot each set of points
    for idx, points in enumerate(trajectories):
        x_values, y_values = zip(*points)
        plt.scatter(
            x_values,
            y_values,
            color=colors[idx],
            label=models[idx],
            marker=markers[idx],
        )

        # Connect the points with lines
        plt.plot(x_values, y_values, color=colors[idx])

        for i, point in enumerate(points):
            plt.annotate(
                str(i),
                xy=point,
                xytext=(-10, 5),
                textcoords='offset points',
                color=colors[idx],
            )

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Contour Plot with Points and Trajectory')
    plt.legend()
    plt.show()

    # Show the plots
    plt.tight_layout()
    plt.show()

    return fig


def create_optimization_plot(
    problem_name, problem, final_results, add_contour_plot=True, colormap='Wistia'
):
    """
    Function that will build out the plots and the solution
    found for the optimization. For our purposes we will mainly
    be interested in the deep learning results.
    """
    # Get the objective function
    objective = problem['objective']

    # Get the bounds for the problem
    x_bounds, y_bounds = problem['bounds']

    # Separate into the minimum and maximum bounds
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    # Some of the problems that we are exploring do not
    # have bounds but we will want to plot them
    if x_max is None:
        x_max = 1e6
    if x_min is None:
        x_min = -1e6
    if y_max is None:
        y_max = 1e6
    if y_min is None:
        y_min = -1e6

    # Create a grid of points
    x = np.linspace(x_min, x_max, 1000)
    y = np.linspace(y_min, y_max, 1000)
    x, y = np.meshgrid(x, y)

    # For the function inputs we need results and a trial
    # Create dummy data
    results = np.zeros((1, 1, 3))
    trial = 0

    # Put objective function in a wrapper
    objective_f = partial(objective, results=results, trial=trial, version='numpy')

    # Create a grid of vectors
    grid = np.stack((x, y), axis=-1)

    # Apply the function on each point of the grid
    z = np.apply_along_axis(objective_f, 2, grid)

    # Create a figure
    fig = plt.figure(figsize=(10, 4.5))

    # Specify 3D plot
    if add_contour_plot:
        ax1 = fig.add_subplot(121, projection='3d')
    else:
        ax1 = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax1.plot_surface(x, y, z, cmap=colormap, alpha=1.0)

    # Add title and labels
    problem_name = problem_name.capitalize()
    ax1.set_title(f'3D Surface Plot of the {problem_name} Function')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel(f'Z ({problem_name} Value)')

    if add_contour_plot:
        # Specify the contour plot
        ax2 = fig.add_subplot(122)

        # Plot the contour
        contour = ax2.contour(x, y, z, cmap=colormap)
        fig.colorbar(contour, ax=ax2)

        # Add the minimum point
        for result in final_results:
            min_x, min_y, _, _, _ = result
            ax2.plot(min_x, min_y, 'ko')  # plot the minimum point as a black dot

        # Add title and labels
        ax2.set_title(f'Contour Plot of the {problem_name} Function')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')

    # Show the plots
    plt.tight_layout()
    plt.show()

    return fig


def get_devices():
    """
    Function to get GPU devices if available. If there are
    no GPU devices available then we will use CPU
    """
    # Default device
    device = torch.device("cpu")

    # Get available GPUs
    n_gpus = torch.cuda.device_count()
    print(f'n_gpus: {n_gpus}')
    if n_gpus > 0:
        gpu_name_list = [f'cuda:{device}' for device in range(n_gpus)]

        # NOTE: For now we will only use the first GPU that is available
        device = torch.device(gpu_name_list[0])

    return device


class DifferentialEvolutionCallback(object):
    def __init__(self):
        self.x_history = []
        self.convergence_history = []

    def record_intermediate_data(self, xk, convergence):
        self.x_history.append(xk)
        self.convergence_history.append(convergence)


class DualAnnealingCallback(object):
    def __init__(self):
        self.x_history = []
        self.f_history = []
        self.context_history = []

    def record_intermediate_data(self, x, f, algorithm_context):
        self.x_history.append(x)
        self.f_history.append(f)
        self.context_history.append(algorithm_context)


class HaltLog:
    """
    Save the iterations from pygranso
    """

    def haltLog(  # noqa
        self,
        iteration,
        x,
        penaltyfn_parts,
        d,
        get_BFGS_state_fn,
        H_regularized,
        ls_evals,
        alpha,
        n_gradients,
        stat_vec,
        stat_val,
        fallback_level,
    ):
        """
        Function that will create the logs from pygranso
        """
        # DON'T CHANGE THIS
        # increment the index/count
        self.index += 1

        # EXAMPLE:
        # store history of x iterates in a preallocated cell array
        self.x_iterates.append(x)
        self.f.append(penaltyfn_parts.f)
        self.tv.append(penaltyfn_parts.tv)
        self.evals.append(ls_evals)

        # keep this false unless you want to implement a custom termination
        # condition
        halt = False
        return halt

    def getLog(self):  # noqa
        """
        Once PyGRANSO has run, you may call this function to get retreive all
        the logging data stored in the shared variables, which is populated
        by haltLog being called on every iteration of PyGRANSO.
        """
        # EXAMPLE
        # return x_iterates, trimmed to correct size
        log = pygransoStruct()
        log.x = self.x_iterates[0 : self.index]
        log.f = self.f[0 : self.index]
        log.tv = self.tv[0 : self.index]
        log.fn_evals = self.evals[0 : self.index]
        return log

    def makeHaltLogFunctions(self, maxit):  # noqa
        """
        Function to make the halt log functions
        """

        # don't change these lambda functions
        def halt_log_fn(  # noqa
            iteration,
            x,
            penaltyfn_parts,
            d,
            get_BFGS_state_fn,
            H_regularized,
            ls_evals,
            alpha,
            n_gradients,
            stat_vec,
            stat_val,
            fallback_level,
        ):
            self.haltLog(
                iteration,
                x,
                penaltyfn_parts,
                d,
                get_BFGS_state_fn,
                H_regularized,
                ls_evals,
                alpha,
                n_gradients,
                stat_vec,
                stat_val,
                fallback_level,
            )

        get_log_fn = lambda: self.getLog()  # noqa

        # Make your shared variables here to store PyGRANSO history data
        # EXAMPLE - store history of iterates x_0,x_1,...,x_k
        self.index = 0
        self.x_iterates = []
        self.f = []
        self.tv = []
        self.evals = []

        # Only modify the body of logIterate(), not its name or arguments.
        # Store whatever data you wish from the current PyGRANSO iteration info,
        # given by the input arguments, into shared variables of
        # makeHaltLogFunctions, so that this data can be retrieved after PyGRANSO
        # has been terminated.
        #
        # DESCRIPTION OF INPUT ARGUMENTS
        #   iter                current iteration number
        #   x                   current iterate x
        #   penaltyfn_parts     struct containing the following
        #       OBJECTIVE AND CONSTRAINTS VALUES
        #       .f              objective value at x
        #       .f_grad         objective gradient at x
        #       .ci             inequality constraint at x
        #       .ci_grad        inequality gradient at x
        #       .ce             equality constraint at x
        #       .ce_grad        equality gradient at x
        #       TOTAL VIOLATION VALUES (inf norm, for determining feasibiliy)
        #       .tvi            total violation of inequality constraints at x
        #       .tve            total violation of equality constraints at x
        #       .tv             total violation of all constraints at x
        #       TOTAL VIOLATION VALUES (one norm, for L1 penalty function)
        #       .tvi_l1         total violation of inequality constraints at x
        #       .tvi_l1_grad    its gradient
        #       .tve_l1         total violation of equality constraints at x
        #       .tve_l1_grad    its gradient
        #       .tv_l1          total violation of all constraints at x
        #       .tv_l1_grad     its gradient
        #       PENALTY FUNCTION VALUES
        #       .p              penalty function value at x
        #       .p_grad         penalty function gradient at x
        #       .mu             current value of the penalty parameter
        #       .feasible_to_tol logical indicating whether x is feasible
        #   d                   search direction
        #   get_BFGS_state_fn   function handle to get the (L)BFGS state data
        #                       FULL MEMORY:
        #                       - returns BFGS inverse Hessian approximation
        #                       LIMITED MEMORY:
        #                       - returns a struct with current L-BFGS state:
        #                           .S          matrix of the BFGS s vectors
        #                           .Y          matrix of the BFGS y vectors
        #                           .rho        row vector of the 1/sty values
        #                           .gamma      H0 scaling factor
        #   H_regularized       regularized version of H
        #                       [] if no regularization was applied to H
        #   fn_evals            number of function evaluations incurred during
        #                       this iteration
        #   alpha               size of accepted size
        #   n_gradients         number of previous gradients used for computing
        #                       the termination QP
        #   stat_vec            stationarity measure vector
        #   stat_val            approximate value of stationarity:
        #                           norm(stat_vec)
        #                       gradients (result of termination QP)
        #   fallback_level      number of strategy needed for a successful step
        #                       to be taken.  See bfgssqpOptionsAdvanced.
        #
        # OUTPUT ARGUMENT
        #   halt                set this to true if you wish optimization to
        #                       be halted at the current iterate.  This can be
        #                       used to create a custom termination condition,
        return halt_log_fn, get_log_fn
