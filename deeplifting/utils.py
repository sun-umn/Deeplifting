# stdlib
import gc
import glob
import os
import random
import uuid
from functools import partial
from typing import Dict

# third party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from pygranso.pygransoStruct import pygransoStruct
from torch.optim.lr_scheduler import OneCycleLR


class PyGransoConfig:
    """
    Configuration class for the pygranso options. The
    options stay fairly consistent and can be used mutiple times.
    The things that we need to set are device, x0 and the maximum
    number of iterations
    """

    def __init__(self, device, x0, max_iterations, printing=False):
        # Set up pygranso structure
        self.opts = pygransoStruct()

        # PyGranso options
        self.opts.x0 = x0
        self.opts.torch_device = device

        # Keep the print freqency at 1 so we can
        # view every iteration
        if printing:
            self.opts.print_frequency = 10

        else:
            # Use this option to turn off printing
            # TODO: May want to observe the t (step size)
            self.opts.print_level = 0

        # LBFGS lookback history size
        self.opts.limited_mem_size = 100

        # Set's the stopping criterion to either L2 or
        # L_inf norm
        self.opts.stat_l2_model = False
        self.opts.double_precision = True

        # Optimization tolerance and stopping criterion
        self.opts.opt_tol = 1e-12

        # Maximum number of iterations will be defined by the
        # problem instance
        self.opts.maxit = max_iterations


class Results:
    """
    Function that combines and concatenates results and saves intermediate
    information during a run.
    TODO: All runs should have the same information so this should be useable
    across all of the different algorithms that we are using.
    """

    def __init__(self, method='deeplifting-pygranso'):
        self.method = method
        self.results = []

    def append_record(
        self,
        global_minimum: float,
        f_init: float,
        f_final: float,
        total_time: float,
        iterations: int,
        fn_evals: int,
        termination_code: int,
        problem_config: Dict[str, int],
        xs: str,
        method: str,
        lr: float,
        objective_values: np.ndarray,
        distance: float,
    ) -> None:
        """
        Utility function that we can use to keep track
        of information that is contained in a record
        """
        solution_tolerance = 1e-4
        # Based on the input information let's compute if the trial
        # was a success or not
        success = int(
            np.abs((global_minimum - f_final) / (global_minimum - f_init))
            <= solution_tolerance
        )

        if self.method in (
            'deeplifting-pygranso',
            'deeplifting-lbfgs',
            'deeplifting-adam',
            'deeplifting-sgd',
        ):
            # Get the neural network configuration information
            self.num_layers = problem_config['num_layers']
            self.num_neurons = problem_config['num_neurons']
            self.lr = problem_config['lr']
            self.input_dimension = problem_config['input_dimension']

            # Save all of the results to a list
            self.results.append(
                (
                    method,
                    xs,
                    global_minimum,
                    np.round(
                        f_init, 6
                    ),  # Seems to be some issue with this at the moment
                    np.round(f_final, 6),
                    total_time,
                    iterations,
                    fn_evals,
                    termination_code,
                    self.num_layers,
                    self.num_neurons,
                    success,
                    lr,
                    objective_values,
                    distance,
                )
            )

        else:
            raise ValueError('Other methods coming soon!')

    def build_and_save_dataframe(self, save_path: str, problem_name: str) -> None:
        """
        Method used to build a dataframe from the current
        data in the lists and save it to a directory.

        user: Input is specific to the directory of the user
        """
        if self.method in (
            'deeplifting-pygranso',
            'deeplifting-lbfgs',
            'deeplifting-adam',
            'deeplifting-sgd',
        ):
            columns = [
                'method',
                'start_position',
                'global_minimum',
                'f_init',
                'f_final',
                'total_time',
                'iterations',
                'fn_evals',
                'termination_code',
                'num_layers',
                'num_neurons',
                'success',
                'learning_rate',
                'objective_values',
                'distance',
            ]

            # Set up the results dataframe
            results_df = pd.DataFrame(self.results, columns=columns)

            # Save the results
            if self.lr is None:
                filename = os.path.join(
                    save_path,
                    f'{problem_name}-relu-{self.num_layers}-{self.num_neurons}-'
                    f'.parquet',
                )
            else:
                filename = os.path.join(
                    save_path,
                    f'{problem_name}-relu-{self.num_layers}-{self.num_neurons}-'
                    f'lr-{self.lr}-input-dim-{self.input_dimension}.parquet',
                )
            results_df.to_parquet(filename)

        else:
            raise ValueError('Other methods coming soon!')


def build_model_complexity_plots(
    path: str, problem: str, weight_initialization: bool, dimension: str
) -> None:
    """
    Function that will take as input a string path and a problem
    name and compile the results to create the success rate vs.
    model complexity plots

    The data should have columns:
    success;
    f_init;
    global_minimum;
    f;
    num_layers;
    units;
    index;
    total_time;
    xs;

    Forgot the termination codes!
    At the moment this assumes low-dimension
    """
    files = os.path.join(path, f'{problem}-relu-*-{weight_initialization}*')
    problem_files = glob.glob(files)
    data = pd.read_parquet(problem_files)

    # Create the results
    results_df = (
        data.groupby(
            ['xs', 'num_layers', 'units']
        )  # Group by each starting point, layers, units  # noqa
        .agg(
            {'success': 'max'}
        )  # This is actual a binary check either there was a success or not  # noqa
        .reset_index()
        .groupby(
            ['num_layers', 'units']
        )  # Now we want to know the percentage of optimization successes  # noqa
        .agg({'success': 'mean'})
        .reset_index()
    )

    # Turn results df into heatmap format
    heatmap_df = results_df.pivot_table(
        index='units', columns='num_layers', values='success'
    )

    # Now create the heatmap with seaborn
    base_save_path = '/panfs/jay/groups/15/jusun/dever120/Deeplifting'
    image_results_path = f'image-results/{dimension}/deeplifting-pygranso/'
    image_save_path = os.path.join(
        base_save_path, image_results_path, f'{problem}-{weight_initialization}.png'
    )

    # Set up figure and create image and save
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    sns.heatmap(
        heatmap_df.sort_index(ascending=False), annot=True, cmap='Spectral_r', ax=ax
    )
    ax.set_title(f'{problem.capitalize()}: Success Rate Vs. Network Complexity')
    ax.set_xlabel('Number of Layers')
    ax.set_ylabel('Number of Neurons Per Layer')

    # Save figure
    fig.savefig(image_save_path)
    plt.close()


def create_unique_experiment_directory():
    """
    Generate a UUID
    """

    unique_id = str(uuid.uuid4())

    # Define the path to the experiments folder and the new UUID directory
    experiments_path = 'experiments'
    new_dir_path = os.path.join(experiments_path, unique_id)

    # Check if the directory already exists
    if not os.path.exists(new_dir_path):
        # Create the directory
        os.makedirs(new_dir_path)
        print(f"Directory {new_dir_path} created.")
    else:
        print(f"Directory {new_dir_path} already exists.")

    # Path for the UUID file under the experiments directory
    uuid_file_path = os.path.join(experiments_path, 'current_experiment_uuid.txt')

    # Write the UUID to the file
    with open(uuid_file_path, 'w') as file:
        file.write(unique_id)
        print(f"UUID {unique_id} written to {uuid_file_path}.")

    # List of main subdirectories
    main_subdirectories = ['low-dimension', 'high-dimension', 'svm']

    # List of subdirectories for low-dimension and high-dimension
    subdirectories = [
        'dual-annealing',
        'differential-evolution',
        'ipopt',
        'pygranso',
        'scip',
        'deeplifting-pygranso',
        'deeplifting-adam',
        'deeplifting-lbfgs',
        'basinhopping',
    ]

    # Create each main subdirectory
    for main_sub in main_subdirectories:
        main_sub_path = os.path.join(new_dir_path, main_sub)
        if not os.path.exists(main_sub_path):
            os.makedirs(main_sub_path)
            print(f"Directory {main_sub_path} created.")

        # Create each subdirectory under the main subdirectory
        for sub in subdirectories:
            sub_path = os.path.join(main_sub_path, sub)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)
                print(f"Directory {sub_path} created.")

    return new_dir_path


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
    """
    Function to initialize a random point of optimization
    for different algorithms
    """
    if bounds is not None:
        # Get the upper and lower bounds
        lower_bounds = bounds['lower_bounds']
        upper_bounds = bounds['upper_bounds']

        # Redefine bounds
        bounds = list(zip(lower_bounds, upper_bounds))

        if len(bounds) != size:
            raise ValueError("The number of bounds must match the size of the vector")

        vector = [np.random.uniform(low, high) for low, high in bounds]
        vector = np.array(vector).flatten()
    else:
        vector = np.random.randn(size)

    return vector


def train_model_to_output(inputs, model, x0, epochs=10000, lr=1e-4, tolerance=1e-3):
    """
    Function takes a model, input tensor, and target output (x0),
    and trains the model's output layer to produce x0 for the given input.

    Parameters:
        inputs: input tensor
        model: model to be trained
        x0: target output tensor
        epochs: number of training epochs (default: 1000)
        lr: learning rate (default: 1e-5)
        tolerance: threshold for L2 distance to stop training (default: 1e-10)
    """

    # # Freeze all layers except the output layer
    # for name, parameters in model.named_parameters():
    #     if (
    #         'alignment_layer' not in name
    #     ):  # assuming 'layer2' is the output layer, adjust if otherwise
    #         parameters.requires_grad = False

    # Begin training
    model.train()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=1,
        pct_start=0.1,
    )
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()  # Zero gradients
        outputs = model(inputs)  # Get model outputs for the input
        outputs = outputs.flatten()  # Flatten the output tensor if needed
        loss = criterion(x0, outputs)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters
        scheduler.step()

        # # Check L2 distance
        # l2_distance = torch.norm(outputs - x0, p=2).item()

    #     # Print loss and L2 distance every 100 epochs
    #     if (epoch + 1) % 1000 == 0:
    #         print(
    #             f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}'
    #             f', L2 Distance: {l2_distance:.4e}'
    #         )

    #     # Check the stopping criterion
    #     if l2_distance < tolerance:
    #         print(
    #             f'Training converged at epoch {epoch+1} with'
    #             f' L2 Distance: {l2_distance:.4e}'
    #         )
    #         break

    # print(f'Final L2 distance {l2_distance:.4e}')

    # # Unfreeze all layers
    # for parameters in model.parameters():
    #     parameters.requires_grad = True

    del (optimizer, scheduler, outputs)
    gc.collect()
    torch.cuda.empty_cache()


def add_jitter(points, jitter_amount=0.05):
    """
    Function that adds a small amount of noise with main
    use case for a scatter plot
    """
    jitter = np.random.uniform(-jitter_amount, jitter_amount, points.shape)
    return points + jitter


def create_contour_plot(problem_name, problem, models, trajectories, colormap='Greys'):
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

    contour = ax1.contourf(x, y, z, levels=10, cmap=colormap, alpha=0.90)
    fig.colorbar(contour, ax=ax1)

    # # Define colors and markers for the models
    # colormaps = [cm.autumn_r, cm.winter, cm.cool, cm.copper, cm.rainbow]
    model_colors = ['red', 'maroon', 'orange', 'black', 'gold']
    markers = ['o', 's', '^', 'd', 'p']

    # Plot each set of points
    for idx, points in enumerate(trajectories):
        if len(points) > 2:
            mid_point = len(points) // 2
            points = [points[0], points[mid_point], points[-1]]

        x_values, y_values = zip(*points)

        # # Create a colormap
        # colormap = colormaps[idx]
        # colors = colormap(np.linspace(0, 1, len(points)))

        plt.scatter(
            x_values,
            y_values,
            color=model_colors[idx],
            label=models[idx],
            marker=markers[idx],
        )

        # for color_idx, (x, y) in enumerate(points):
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]

            # Connect the points with lines
            # plt.plot(x, y, markers[idx], color=colors[i])
            plt.plot([x1, x2], [y1, y2], color=model_colors[idx], lw=3)

            # Add the annotation
            plt.annotate(
                str(i),
                xy=(x1, y1),
                xytext=(-10, 5),
                textcoords='offset points',
                color=model_colors[idx],
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
    problem_name, problem, add_contour_plot=False, colormap='Wistia'
):
    """
    Function that will build out the plots and the solution
    found for the optimization. For our purposes we will mainly
    be interested in the deep learning results.
    """
    # Get the objective function
    objective = problem['objective']

    # Get the bounds for the problem

    # Get the upper bounds & lower bounds
    lower_bounds = problem['bounds']['lower_bounds']
    upper_bounds = problem['bounds']['upper_bounds']

    x_min, y_min = lower_bounds
    x_max, y_max = upper_bounds

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

    # Put objective function in a wrapper
    objective_f = partial(objective, version='numpy')

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


class BasinHoppingCallback:
    """
    Callback to save intermediate results for
    Basinhopping
    """

    def __init__(self):
        self.x_history = []
        self.f_history = []
        self.accecpt_history = []

    def record_intermediate_data(self, x, f, accept):
        self.x_history.append(x)
        self.f_history.append(f)
        self.accecpt_history.append(accept)


class DifferentialEvolutionCallback:
    """
    Callback to save intermediate results for
    Differential Evolution
    """

    def __init__(self):
        self.x_history = []
        self.f_history = []
        self.convergence_history = []

    def record_intermediate_data(self, intermediate_result):
        # Get the values from OptimizedResult
        x = intermediate_result.x
        f = intermediate_result.fun
        convergence = intermediate_result.convergence

        # Now append the values to the list
        self.x_history.append(x)
        self.f_history.append(f)
        self.convergence_history.append(convergence)


class DualAnnealingCallback:
    """
    Callback to save intermediate results for
    Dual Annealing
    """

    def __init__(self):
        self.x_history = []
        self.f_history = []
        self.context_history = []

    def record_intermediate_data(self, x, f, algorithm_context):
        self.x_history.append(x)
        self.f_history.append(f)
        self.context_history.append(algorithm_context)


class IPOPTCallback:
    """
    Callback to save intermediate results for
    IPOPT
    """

    def __init__(self):
        self.x_history = []
        self.f_history = []

    def record_intermediate_data(
        self,
        alg_mod,
        iter_count,
        obj_value,
        inf_pr,
        inf_du,
        mu,
        d_norm,
        regularization_size,
        alpha_du,
        alpha_pr,
    ):
        pass


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
        # # store history of x iterates in a preallocated cell array
        # self.x_iterates.append(x)
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
        # log.x = self.x_iterates[0 : self.index]
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
        # self.x_iterates = []
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
