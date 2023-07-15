# stdlib
from functools import partial

# third party
import matplotlib.pyplot as plt
import numpy as np
import torch


def create_optimization_plot(problem_name, problem, final_results, colormap='Wistia'):
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
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
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
    ax1 = fig.add_subplot(121, projection='3d')

    # Plot the surface
    ax1.plot_surface(x, y, z, cmap=colormap, alpha=1.0)

    # Add title and labels
    problem_name = problem_name.capitalize()
    ax1.set_title(f'3D Surface Plot of the {problem_name} Function')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel(f'Z ({problem_name} Value)')

    # Specify the contour plot
    ax2 = fig.add_subplot(122)

    # Plot the contour
    contour = ax2.contourf(x, y, z, cmap=colormap)
    fig.colorbar(contour, ax=ax2)

    # Add the minimum point
    for result in final_results:
        min_x, min_y, _, _ = result
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
    if n_gpus > 0:
        gpu_name_list = [f'cuda:{device}' for device in range(n_gpus)]

        # NOTE: For now we will only use the first GPU that is available
        device = torch.device(gpu_name_list[0])

    return device
