# stdlib
import gc
import time
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
from deeplifting.models import DeepliftingSkipMLP
from deeplifting.utils import get_devices, set_seed


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
        start_time = time.time()
        result = minimize_ipopt(fn, x0, bounds=bounds)
        end_time = time.time()
        total_time = end_time - start_time

        x_tuple = tuple(x for x in result.x)
        fn_values.append(x_tuple + (result.fun, 'IPOPT', total_time))

    return {'results': results, 'final_results': fn_values}


def run_dual_annealing(
    problem: Dict, trials: int, init_temp=5230, res_temp=2e-5, vis=2.62, acpt=-5.0
):
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

    # Some of the problems may be unbounded but dual annealing
    # and differential evolution need to have bounds provided
    updated_bounds = []
    for constr in bounds:
        a, b = constr
        if a is None:
            a = -1e6
        if b is None:
            b = 1e6
        updated_bounds.append((a, b))

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

        # Get the result
        start_time = time.time()
        result = dual_annealing(
            fn,
            updated_bounds,
            maxiter=100,
            initial_temp=init_temp,
            restart_temp_ratio=res_temp,
            visit=vis,
            accept=acpt,
        )
        end_time = time.time()
        total_time = end_time - start_time
        x_tuple = tuple(x for x in result.x)
        fn_values.append(x_tuple + (result.fun, 'Dual Annealing', total_time))

    return {'results': results, 'final_results': fn_values}


def run_differential_evolution(
    problem: Dict, trials: int, strat='best1bin', mut=0.5, recomb=0.7
):
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

    # Some of the problems may be unbounded but dual annealing
    # and differential evolution need to have bounds provided
    updated_bounds = []
    near_zero_bounds = []
    for constr in bounds:
        a, b = constr
        if a is None:
            a = float(-1e20)
        if b is None:
            b = float(1e20)
        updated_bounds.append((a, b))

    # Need to modify x0 for problems like ex8_6_2
    for index, constr in enumerate(updated_bounds):
        a, b = constr
        if (a == -1e-6) or (b == 1e-6):
            near_zero_bounds.append(index)

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

        # Get the result
        start_time = time.time()
        result = differential_evolution(
            fn,
            updated_bounds,
            maxiter=100,
            strategy=strat,
            mutation=mut,
            recombination=recomb,
        )
        end_time = time.time()
        total_time = end_time - start_time
        x_tuple = tuple(x for x in result.x)
        fn_values.append(x_tuple + (result.fun, 'Differential Evolution', total_time))

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
    # In some cases all of the bounds will be
    # None so we will need to set the ci struct
    # to None
    ci = None

    if np.any(np.array(bounds).flatten() != None):  # noqa
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

        # Get the bounds of the problem
        if dimensions <= 2:
            bounds = problem['bounds']
        else:
            bounds = problem['bounds']

            if len(bounds) > 1:
                bounds = bounds

            else:
                bounds = bounds * dimensions

        # Create x0 based on the bounds
        x0_values = []
        for bound in bounds:
            a, b = bound
            if a is None and b is None:
                x_value = torch.randn(1) * 1e3

            elif a is None and b is not None:
                x_value = torch.randn(1) * 1e3
                x_value = torch.clamp(x_value, min=None, max=b)

            elif a is not None and b is None:
                x_value = torch.randn(1) * 1e3
                x_value = torch.clamp(x_value, min=a, max=None)

            else:
                x_value = a + (b - a) / 2.0 * (torch.sin(torch.randn(1)) + 1.0)

            x0_values.append(x_value)

        x0 = torch.tensor(x0_values).reshape(dimensions, 1)
        x0 = x0.to(device=device, dtype=torch.double)

        # Pygranso options
        opts.x0 = x0
        opts.torch_device = device
        # opts.print_frequency = 0
        opts.print_level = 0
        opts.limited_mem_size = 50
        opts.stat_l2_model = False
        opts.double_precision = True
        opts.viol_ineq_tol = 1e-10
        opts.opt_tol = 1e-10
        opts.maxit = 2000

        # Objective function
        objective = problem['objective']

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
        start_time = time.time()
        soln = pygranso(var_spec=var_in, combined_fn=comb_fn, user_opts=opts)
        end_time = time.time()
        total_time = end_time - start_time

        # Get final x we will also need to map
        # it to the same bounds
        x = soln.final.x.cpu().numpy().flatten()

        f = soln.final.f
        b = soln.best.f

        x_tuple = tuple(x)
        fn_values.append(x_tuple + (f, 'PyGRANSO', total_time))

        del (var_in, x0, opts, soln, x, f, b)
        gc.collect()
        torch.cuda.empty_cache()

    return {'results': results, 'final_results': fn_values}


def deeplifting_predictions(outputs1, outputs2, objective, bounds):
    """
    Function to create the outputs for the
    deeplifting framework
    """
    # Let's try out trick from topology
    # optimization instead of relying on the
    # inequality constraint
    # If we map x and y to [0, 1] and then shift
    # the interval we can accomplist the same
    # thing we can use a + (b - a) * x

    # For sin [-1, 1]
    # c + (d - c) / (b - a) * (x - a)
    # c + (d - c) / (2) * (x + 1)

    # Try updating the way we define the bounds
    x_values_float = []
    for index, cnstr in enumerate(bounds):
        a, b = cnstr
        if (a is None) and (b is None):
            x_constr = outputs1[index]
        elif (a is None) or (b is None):
            x_constr = torch.clamp(outputs1[index], min=a, max=b)

        # Being very explicit about this condition just in case
        # to avoid weird behavior
        elif (a is not None) and (b is not None):
            x_constr = a + (b - a) / 2.0 * (torch.sin(outputs1[:, index]) + 1)
            # x_constr = a + (b - a) * torch.sigmoid(outputs2[:, index] * 6)
        x_values_float.append(x_constr)

    x_float = torch.stack(x_values_float, axis=1)

    # # Integer outputs
    # x_values_trunc = []
    # for index, cnstr in enumerate(bounds):
    #     a, b = cnstr
    #     if (a is None) and (b is None):
    #         x_constr = outputs2[index]
    #     elif (a is None) or (b is None):
    #         x_constr = torch.clamp(outputs2[index], min=a, max=b)

    #     # Being very explicit about this condition just in case
    #     # to avoid weird behavior
    #     elif (a is not None) and (b is not None):
    #         # x_constr = a + (b - a) / 2.0 * (torch.sin(outputs2[:, index]) + 1)
    #         x_constr = a + (b - a) * torch.sigmoid(outputs2[:, index] * 6)
    #     x_values_trunc.append(x_constr)

    # x_trunc = torch.stack(x_values_trunc, axis=1)
    # # x_trunc = torch.trunc(x_trunc)

    # x = torch.vstack((x_float, x_trunc))
    x = x_float

    # Iterate over the objective values
    objective_values = []
    for i in range(len(x)):
        f = objective(x[i, :])
        objective_values.append(f)

    objective_values = torch.stack(objective_values)
    f = torch.min(objective_values)

    # Need to get the minimum of f
    idx_min = torch.argmin(objective_values)
    x = x[idx_min, :]
    x = x.detach().cpu().numpy().flatten()

    return x, f


def deeplifting_nd_fn(model, objective, bounds):
    """
    Combined funtion used for PyGranso
    """
    outputs1, outputs2 = model(inputs=None)

    # Get x1 and x2 so we can add the bounds
    # outputs = torch.sigmoid(outputs)
    # x = outputs.mean(axis=0)
    # print(f'Output x {x}')
    x, f = deeplifting_predictions(outputs1, outputs2, objective, bounds)

    # Inequality constraint
    ci = None

    # Equality constraing
    ce = None

    return f, ci, ce


def run_deeplifting(
    problem: Dict,
    trials: int,
    input_size=512,
    hidden_sizes=(512, 512, 512),
    activation='sine',
    agg_function='sum',
):
    """
    Function that runs our preimer method of deeplifting.
    The idea here is to reparmeterize an optimization objective
    so we can have better optimization and convergence
    """
    # Get the device (CPU for now)
    dimensions = problem['dimensions']
    device = get_devices()
    print(device)
    fn_values = []

    for trial in range(trials):
        # Deeplifting model with skip connections
        model = DeepliftingSkipMLP(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=dimensions,
            skip_every_n=1,
            activation=activation,
            agg_function=agg_function,
            seed=trial,
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
        # opts.print_level = 0
        opts.limited_mem_size = 5
        opts.stat_l2_model = False
        opts.double_precision = True
        # opts.viol_ineq_tol = 1e-15
        opts.opt_tol = 1e-20
        opts.maxit = 1000

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
        comb_fn = lambda model: deeplifting_nd_fn(model, fn, bounds)  # noqa

        # Run the main algorithm
        start_time = time.time()
        soln = pygranso(var_spec=model, combined_fn=comb_fn, user_opts=opts)
        end_time = time.time()
        total_time = end_time - start_time

        # Get final x we will also need to map
        # it to the same bounds
        outputs1, outputs2 = model(inputs=None)
        final_results = np.zeros((1, 1, 3))
        final_fn = lambda x: objective(
            x, results=final_results, trial=0, version='pytorch'
        )
        xf, f = deeplifting_predictions(outputs1, outputs2, final_fn, bounds)
        data_point = tuple(xf) + (
            float(f.detach().cpu().numpy()),
            'Deeplifting',
            total_time,
        )
        fn_values.append(data_point)

        # Collect garbage and empty cache
        del (model, nvar, x0, opts, soln, outputs1, outputs2, xf, f)
        gc.collect()
        torch.cuda.empty_cache()

    return {'results': results, 'final_results': fn_values}


def run_optimization(problem: Dict, algorithms: List) -> pd.DataFrame:
    """
    Function that runs optimization with different specified
    algorithms for our deeplifitng research.
    """
