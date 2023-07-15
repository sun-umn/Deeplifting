# stdlib
import gc
import random
from typing import Dict, List

# third party
import numpy as np
import pandas as pd
import torch
from cyipopt import minimize_ipopt
from pygranso.private.getNvar import getNvarTorch
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from scipy.optimize import differential_evolution, dual_annealing

# first party
from deeplifting.models import DeepliftingMLP
from deeplifting.utils import get_devices


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


def run_ipopt(problem: Dict, trials: int):
    """
    Function that runs IPOPT on one of our test
    functions for deeplifting.

    TODO: In the documentation there are ways to improve
    IPOPT by using the gradient, jacobian and hessian
    information we will implement that with jax
    but may need to rework some of our test functions
    """
    # Objective function
    objective = problem['objective']

    # Get the maximum iterations
    max_iterations = problem['max_iterations']

    # Get dimensions of the problem
    dimensions = problem['dimensions']

    # Get the bounds of the problem
    if dimensions <= 2:
        bounds = problem['bounds']
    else:
        bounds = problem['bounds']

        if len(bounds) > 1:
            bounds = bounds

        else:
            bounds = bounds * dimensions

    # Save the results
    # We will store the optimization steps here
    results = np.zeros((trials, max_iterations, dimensions + 1)) * np.nan
    fn_values = []

    for trial in range(trials):
        # Set the random seed
        set_seed(trial)

        # Initial guess (starting point for IPOPT)
        x0 = np.random.rand(dimensions)

        # Get the objective
        fn = lambda x: objective(  # noqa
            x, results=results, trial=trial, version='numpy'
        )  # noqa

        # Call IPOPT
        result = minimize_ipopt(fn, x0, bounds=bounds)
        x_tuple = tuple(x for x in result.x)
        fn_values.append(x_tuple + (result.fun,))

    return {'results': results, 'final_results': fn_values}


def run_dual_annealing(problem: Dict, trials: int):
    """
    Function that runs dual annealing for a
    specified optimization problem
    """
    # Objective function
    objective = problem['objective']

    # Get the number of dimensions
    dimensions = problem['dimensions']

    # Get the bounds of the problem
    if dimensions <= 2:
        bounds = problem['bounds']
    else:
        bounds = problem['bounds']

        if len(bounds) > 1:
            bounds = bounds

        else:
            bounds = bounds * dimensions

    # Get the maximum iterations
    max_iterations = problem['max_iterations']

    # Save the results
    # We will store the optimization steps here
    results = np.zeros((trials, max_iterations, dimensions + 1)) * np.nan
    fn_values = []

    for trial in range(trials):
        # Set a random seed
        np.random.seed(trial)

        # Set up the function with the results
        fn = lambda x: objective(  # noqa
            x, results=results, trial=trial, version='numpy'
        )

        # Random starting point for dual annealing
        x0 = np.random.rand(dimensions)

        # Get the result
        result = dual_annealing(fn, bounds, x0=x0, maxiter=max_iterations)
        x_tuple = tuple(x for x in result.x)
        fn_values.append(x_tuple + (result.fun,))

    return {'results': results, 'final_results': fn_values}


def run_differential_evolution(problem: Dict, trials: int):
    """
    Function that runs differential evolution for a
    specified optimization problem
    """
    # Objective function
    objective = problem['objective']

    # Get the number of dimensions
    dimensions = problem['dimensions']

    # Get the bounds of the problem
    if dimensions <= 2:
        bounds = problem['bounds']
    else:
        bounds = problem['bounds']

        if len(bounds) > 1:
            bounds = bounds

        else:
            bounds = bounds * dimensions

    # Get the maximum iterations
    max_iterations = problem['max_iterations']

    # Save the results
    # We will store the optimization steps here
    results = np.zeros((trials, max_iterations, dimensions + 1)) * np.nan
    fn_values = []

    for trial in range(trials):
        # Set a random seed
        np.random.seed(trial)

        # Set up the function with the results
        fn = lambda x: objective(  # noqa
            x, results=results, trial=trial, version='numpy'
        )

        # Random starting point for dual annealing
        x0 = np.random.rand(dimensions)

        # Get the result
        result = differential_evolution(fn, bounds, x0=x0, maxiter=max_iterations)
        x_tuple = tuple(x for x in result.x)
        fn_values.append(x_tuple + (result.fun,))

    return {'results': results, 'final_results': fn_values}


def pygranso_fn(X_struct, objective, bounds):
    """
    Function to run PyGranso without a neural
    network approximation to the inputs
    """
    x1, x2 = X_struct.x1, X_struct.x2
    x = torch.cat((x1, x2))
    f = objective(x)

    # Setup the bounds for the inequality
    # a <= x <= b
    # -> a - x <= 0
    # -> x - b <= 0
    x1_bounds, x2_bounds = bounds
    a1, b1 = x1_bounds
    a2, b2 = x2_bounds

    c1 = a1 - x1
    c2 = x1 - b1
    c3 = a2 - x2
    c4 = x2 - b2

    # Inequality constraints
    ci = pygransoStruct()
    ci.c1 = c1
    ci.c2 = c2
    ci.c3 = c3
    ci.c4 = c4

    # Equality constraints
    ce = None

    return f, ci, ce


def pygranso_nd_fn(X_struct, objective, bounds):
    """
    Function to run PyGranso without a neural
    network approximation to the inputs
    """
    x_values = []
    for key, value in X_struct.__dict__.items():
        x_values.append(value)
    x = torch.cat(x_values)
    f = objective(x)

    # Setup the bounds for the inequality
    # a <= x <= b
    # -> a - x <= 0
    # -> x - b <= 0
    # Inequality constraints
    ci = pygransoStruct()
    for index, cnstr in enumerate(bounds):
        a, b = cnstr
        if a is None and b is None:
            pass
        elif a is None:
            setattr(ci, f'ca{index}', a - x[index])
        elif b is None:
            setattr(ci, f'cb{index}', x[index] - b)
        else:
            setattr(ci, f'ca{index}', a - x[index])
            setattr(ci, f'cb{index}', x[index] - b)

    # Equality constraints
    ce = None

    return f, ci, ce


def run_pygranso(problem: Dict, trials: int):
    """
    Function that runs PyGranso for our deeplifting
    comparisions
    """
    # Get the device (CPU for now)
    device = get_devices()
    fn_values = []

    for trial in range(trials):
        # Seed everything
        set_seed(trial)

        # Get the number of dimensions for the problem
        dimensions = problem['dimensions']

        # var in
        var_in = {}
        for index in range(dimensions):
            var_in[f'x{index + 1}'] = [1]

        # Setup a pygransoStruct for the algorithm
        # options
        opts = pygransoStruct()

        # Inital x0
        x0 = torch.rand(size=(dimensions, 1), device=device, dtype=torch.double)

        opts.x0 = x0
        opts.torch_device = device
        opts.print_level = 1
        opts.limited_mem_size = 150
        opts.stat_l2_model = False
        opts.double_precision = True
        opts.viol_ineq_tol = 1e-10
        opts.opt_tol = 1e-10
        opts.maxit = 2000

        # Objective function
        objective = problem['objective']

        # Get the bounds of the problem
        if dimensions <= 2:
            bounds = problem['bounds']
        else:
            bounds = problem['bounds']

            if len(bounds) > 1:
                bounds = bounds

            else:
                bounds = bounds * dimensions

        # Get the maximum iterations
        max_iterations = problem['max_iterations']

        # results
        results = np.zeros((trials, max_iterations, dimensions + 1)) * np.nan

        # Set up the function with the results
        fn = lambda x: objective(  # noqa
            x, results=results, trial=trial, version='pytorch'
        )

        # Combined function
        comb_fn = lambda x: pygranso_nd_fn(x, fn, bounds)  # noqa

        # Run the main algorithm
        soln = pygranso(var_spec=var_in, combined_fn=comb_fn, user_opts=opts)

        # Get final x we will also need to map
        # it to the same bounds
        x = soln.final.x.cpu().numpy().flatten()

        f = soln.final.f
        b = soln.best.f

        x_tuple = tuple(x)
        fn_values.append(x_tuple + (f,) + (b,))

        del (var_in, x0, opts, soln, x, f, b)
        gc.collect()
        torch.cuda.empty_cache()

    return {'results': results, 'final_results': fn_values}


def deeplifting_fn(model, objective, bounds):
    """
    Combined funtion used for PyGranso
    """
    outputs = model(inputs=None)

    # Get x1 and x2 so we can add the bounds
    outputs = torch.sigmoid(outputs)
    x1, x2 = outputs.mean(axis=0)

    # Let's try out trick from topology
    # optimization instead of relying on the
    # inequality constraint
    # If we map x and y to [0, 1] and then shift
    # the interval we can accomplist the same
    # thing we can use a + (b - a) * x
    # Get first bounds
    a1, b1 = bounds[0]
    x1 = a1 + (b1 - a1) * x1

    # Get second bounds
    a2, b2 = bounds[1]
    x2 = a2 + (b2 - a2) * x2

    # Objective function
    x = torch.stack((x1, x2))
    f = objective(x)

    # Inequality constraint
    ci = None

    # Equality constraing
    ce = None

    return f, ci, ce


def deeplifting_nd_fn(model, objective, bounds):
    """
    Combined funtion used for PyGranso
    """
    outputs = model(inputs=None)

    # Get x1 and x2 so we can add the bounds
    # outputs = torch.sigmoid(outputs)
    x = outputs.mean(axis=0)

    # Let's try out trick from topology
    # optimization instead of relying on the
    # inequality constraint
    # If we map x and y to [0, 1] and then shift
    # the interval we can accomplist the same
    # thing we can use a + (b - a) * x

    # Try updating the way we define the bounds
    x_values = []
    for index, cnstr in enumerate(bounds):
        a, b = cnstr
        if (a is None) and (b is None):
            x_constr = x[index]
        elif (a is None) or (b is None):
            x_constr = torch.clamp(x[index], min=a, max=b)

        # Being very explicit about this condition just in case
        # to avoid weird behavior
        elif (a is not None) and (b is not None):
            x_constr = a + (b - a) * torch.sigmoid(x[index])
        x_values.append(x_constr)

    x = torch.stack(x_values)
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
    dimensions = problem['dimensions']
    device = get_devices()
    fn_values = []

    for trial in range(trials):
        # Seed everything
        set_seed(trial)

        model = DeepliftingMLP(
            input_size=25,
            # layer_sizes=(256, 256, 256, 256, 256),
            layer_sizes=(512, 512, 512, 512),
            # layer_sizes=(1024, 512, 512, 512, 1024),
            output_size=dimensions,
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
        opts.print_level = 1
        opts.limited_mem_size = 150
        opts.stat_l2_model = False
        opts.double_precision = True
        opts.viol_ineq_tol = 1e-10
        opts.opt_tol = 1e-10
        opts.maxit = 2000

        # Objective function
        objective = problem['objective']

        # Get the bounds of the problem
        if dimensions <= 2:
            bounds = problem['bounds']
        else:
            bounds = problem['bounds']

            if len(bounds) > 1:
                bounds = bounds

            else:
                bounds = bounds * dimensions

        # Get the maximum iterations
        max_iterations = problem['max_iterations']

        # results
        results = np.zeros((trials, max_iterations, dimensions + 1)) * np.nan

        # Set up the function with the results
        fn = lambda x: objective(  # noqa
            x, results=results, trial=trial, version='pytorch'
        )

        # # Combined function
        # if dimensions <= 2:
        #     comb_fn = lambda model: deeplifting_fn(model, fn, bounds)  # noqa
        # else:
        comb_fn = lambda model: deeplifting_nd_fn(model, fn, bounds)  # noqa

        # Run the main algorithm
        soln = pygranso(var_spec=model, combined_fn=comb_fn, user_opts=opts)

        # Get final x we will also need to map
        # it to the same bounds
        outputs = model(inputs=None)
        outputs = torch.sigmoid(outputs)
        x = outputs.mean(axis=0)

        xf = []
        for idx, cnstr in enumerate(bounds):
            a, b = cnstr
            if a is None and b is None:
                x_constr = x[idx]
                x_constr = float(x_constr.detach().cpu().numpy())
            elif a is None or b is None:
                x_constr = torch.clamp(x[idx], min=a, max=b)
                x_constr = float(x_constr.detach().cpu().numpy())
            else:
                x_constr = a + (b - a) * x[idx].detach().cpu().numpy()
            xf.append(x_constr)

        f = soln.final.f
        b = soln.best.f
        data_point = tuple(xf) + (f,) + (b,)
        fn_values.append(data_point)

        # Collect garbage and empty cache
        del (model, nvar, x0, opts, soln, outputs, x, xf, f, b)
        gc.collect()
        torch.cuda.empty_cache()

    return {'results': results, 'final_results': fn_values}


def run_optimization(problem: Dict, algorithms: List) -> pd.DataFrame:
    """
    Function that runs optimization with different specified
    algorithms for our deeplifitng research.
    """
