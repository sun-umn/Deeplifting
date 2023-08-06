# stdlib
import random
from functools import partial

# third party
import matplotlib.pyplot as plt
import numpy as np
import torch


class DualAnnealingCallback(object):
    def __init__(self):
        self.x_history = []
        self.f_history = []
        self.context_history = []

    def record_intermediate_data(self, x, f, algorithm_context):
        self.x_history.append(x)
        self.f_history.append(f)
        self.context_history.append(algorithm_context)


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
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)

    contour = ax1.contourf(x, y, z, levels=10, cmap=colormap)
    fig.colorbar(contour, ax=ax1)

    # Define colors and markers for the models
    colors = ['grey', 'maroon', 'black']

    # Plot each set of points
    for idx, points in enumerate(trajectories):
        points_with_jitter = add_jitter(np.array(points))
        x_values, y_values = zip(*points_with_jitter)
        plt.scatter(
            x_values,
            y_values,
            color=colors[idx],
            label=models[idx],
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
        contour = ax2.contourf(x, y, z, cmap=colormap)
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
