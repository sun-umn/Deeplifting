# stdlib
from typing import Dict, List

# third party
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, dual_annealing


def run_dual_annealing(problem: Dict, trials: int):
    """
    Function that runs dual annealing for a
    specified optimization problem
    """
    # Objective function
    objective = problem['objective']

    # Get the bounds of the problem
    bounds = problem['bounds']

    # Get the maximum iterations
    max_iterations = problem['max_iterations']

    # Save the results
    # We will store the optimization steps here
    results = np.zeros((trials, max_iterations, 3)) * np.nan
    fn_values = []

    for trial in range(trials):
        # Set a random seed
        np.random.seed(trial)

        # Set up the function with the results
        fn = lambda x: objective(
            x, results=results, trial=trial, version='numpy'
        )  # noqa

        # Get the result
        result = dual_annealing(fn, bounds, maxiter=max_iterations)
        fn_values.append((result.x, result.fun))

    return {'results': results, 'final_results': fn_values}


def run_differential_evolution(problem: Dict, trials: int):
    """
    Function that runs differential evolution for a
    specified optimization problem
    """
    # Objective function
    objective = problem['objective']

    # Get the bounds of the problem
    bounds = problem['bounds']

    # Get the maximum iterations
    max_iterations = problem['max_iterations']

    # Save the results
    # We will store the optimization steps here
    results = np.zeros((trials, max_iterations, 3)) * np.nan
    fn_values = []

    for trial in range(trials):
        # Set a random seed
        np.random.seed(trial)

        # Set up the function with the results
        fn = lambda x: objective(
            x, results=results, trial=trial, version='numpy'
        )  # noqa

        # Get the result
        result = differential_evolution(fn, bounds, maxiter=max_iterations)
        fn_values.append((result.x, result.fun))

    return {'results': results, 'final_results': fn_values}


def run_optimization(problem: Dict, algorithms: List) -> pd.DataFrame:
    """
    Function that runs optimization with different specified
    algorithms for our deeplifitng research.
    """
    pass
