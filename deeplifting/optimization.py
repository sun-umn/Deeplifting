# stdlib
import random
from typing import Dict, List

# third party
import numpy as np
import pandas as pd
import torch
from pygranso.private.getNvar import getNvarTorch
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from scipy.optimize import differential_evolution, dual_annealing

# first party
from deeplifting.models import DeepliftingMLP


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
        fn = lambda x: objective(  # noqa
            x, results=results, trial=trial, version='numpy'
        )

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
        fn = lambda x: objective(  # noqa
            x, results=results, trial=trial, version='numpy'
        )

        # Get the result
        result = differential_evolution(fn, bounds, maxiter=max_iterations)
        fn_values.append((result.x, result.fun))

    return {'results': results, 'final_results': fn_values}


def deeplifting_fn(model, objective, bounds):
    """
    Combined funtion used for PyGranso
    """
    outputs = model(inputs=None)

    # Get x1 and x2 so we can add the bounds
    x1, x2 = outputs

    # Let's try out trick from topology
    # optimization instead of relying on the
    # inequality constraint
    # If we map x and y to [0, 1] and then shift
    # the interval we can accomplist the same
    # thing we can use a + (b - a) * x
    # Get first bounds
    a1, b1 = bounds[0]
    x1 = a1 + (b1 - a1) * torch.sigmoid(x1)

    # Get second bounds
    a2, b2 = bounds[1]
    x2 = a2 + (b2 - a2) * torch.sigmoid(x2)

    # Objective function
    x = torch.stack((x1, x2))
    f = objective(x)

    # Inequality constraint
    ci = None

    # Equality constraing
    ce = None

    return f, ci, ce


def run_deeplifting(problem: Dict, trials: int):
    """
    Function that runs our preimer method of deeplifting.
    The idea here is to reparmeterize an optimization objective
    so we can have better optimization and convergence
    """
    # Get the device (CPU for now)
    device = torch.device('cpu')
    fn_values = []

    for trial in range(trials):
        # Seed everything
        set_seed(trial)

        # Define the model
        model = DeepliftingMLP(
            input_size=100, layer_sizes=(128, 256, 512, 256, 128), output_size=2
        )
        model = model.to(device=device, dtype=torch.double)
        nvar = getNvarTorch(model.parameters())
        # Setup a pygransoStruct for the algorithm
        # options
        opts = pygransoStruct()

        # Inital x0
        x0 = (
            torch.nn.utils.parameters_to_vector(model.parameters())
            .detach()
            .reshape(nvar, 1)
            .to(device=device, dtype=torch.double)
        )

        opts.x0 = x0
        opts.torch_device = device
        opts.print_frequency = 1
        opts.limited_mem_size = 25
        opts.stat_l2_model = False
        opts.double_precision = True
        opts.viol_ineq_tol = 1e-10
        opts.opt_tol = 1e-10

        # Objective function
        objective = problem['objective']

        # Get the bounds of the problem
        bounds = problem['bounds']

        # Get the maximum iterations
        max_iterations = problem['max_iterations']

        # results
        results = np.zeros((trials, max_iterations, 3)) * np.nan

        # Set up the function with the results
        fn = lambda x: objective(x, results=results, trial=0, version='pytorch')  # noqa

        # Combined function
        comb_fn = lambda model: deeplifting_fn(model, fn, bounds)  # noqa

        # Run the main algorithm
        soln = pygranso(var_spec=model, combined_fn=comb_fn, user_opts=opts)

        # Get final x
        x1, x2 = model(inputs=None)
        f = soln.f
        fn_values.append((x1, x2, f))

    return {'results': results, 'final_results': fn_values}


def run_optimization(problem: Dict, algorithms: List) -> pd.DataFrame:
    """
    Function that runs optimization with different specified
    algorithms for our deeplifitng research.
    """
