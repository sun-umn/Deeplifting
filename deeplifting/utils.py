# stdlib
from functools import partial

# third party
import matplotlib.pyplot as plt
import numpy as np


def create_optimization_plot(problem, final_results):
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
    ax1.plot_surface(x, y, z, cmap='Wistia', alpha=1.0)

    # Add title and labels
    ax1.set_title('3D Surface Plot of the Ackley Function')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z (Ackley Value)')

    # Specify the contour plot
    ax2 = fig.add_subplot(122)

    # Plot the contour
    contour = ax2.contourf(x, y, z, cmap='Wistia')
    fig.colorbar(contour, ax=ax2)

    # Add the minimum point
    for result in final_results:
        min_x, min_y, _, _ = result
        ax2.plot(min_x, min_y, 'ko')  # plot the minimum point as a black dot

    # Add title and labels
    ax2.set_title('Contour Plot of the Ackley Function')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    # Show the plots
    plt.tight_layout()
    plt.show()

    return fig
