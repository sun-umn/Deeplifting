# stdlib
from typing import Dict, List

# third party
import numpy as np
import pandas as pd
from scipy.optimize import dual_annealing


def run_dual_annealing(problem: Dict, trials: int):
    """
    Function that runs dual annealing for a
    specified optimization problem
    """
    # Objective function
    fn = problem['objective']

    # Get the bounds of the problem
    bounds = problem['bounds']

    # Get the maximum iterations
    max_iterations = problem['max_iterations']

    # Save the results
    # We will store the optimization steps here
    _ = np.zeros((max_iterations, trials))

    for trial in range(trials):
        # Set a random seed
        np.random.seed(trial)

        # Get the result
        _ = dual_annealing(fn, bounds, maxiter=max_iterations)

        # TODO: Save the results


def run_optimization(problem: Dict, algorithms: List) -> pd.DataFrame:
    """
    Function that runs optimization with different specified
    algorithms for our deeplifitng research.
    """
    pass
