# stdlib
import gc
import json
import os
import time
from typing import Any, Dict, List

# third party
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from cyipopt import minimize_ipopt
from pygranso.private.getNvar import getNvarTorch
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from pyomo.environ import ConcreteModel, Objective, SolverFactory, Var, minimize
from scipy.optimize import basinhopping, differential_evolution, dual_annealing
from torch.optim.lr_scheduler import OneCycleLR

# first party
from deeplifting.models import DeepliftingSkipMLP
from deeplifting.problems import HIGH_DIMENSIONAL_PROBLEMS_BY_NAME
from deeplifting.utils import (
    DifferentialEvolutionCallback,
    DualAnnealingCallback,
    HaltLog,
    get_devices,
    initialize_vector,
    set_seed,
    train_model_to_output,
)

# from torch.optim.lr_scheduler import ReduceLROnPlateau


# Do NOT save intermediate paths for high dimensional problems
EXCLUDE_PROBLEMS = list(HIGH_DIMENSIONAL_PROBLEMS_BY_NAME.keys())


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
        # TODO: Need to provide a better starting point here
        x0 = initialize_vector(size=dimensions, bounds=bounds)

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
    results = np.zeros((trials, max_iterations * 10, dimensions + 1)) * np.nan
    fn_values = []
    callbacks = []

    for trial in range(trials):
        # Set a random seed
        np.random.seed(trial)

        # Initial point
        x0 = initialize_vector(size=dimensions, bounds=bounds)

        # Set up the function with the results
        fn = lambda x: objective(  # noqa
            x, results=results, trial=trial, version='numpy'
        )

        # Callback
        callback = DualAnnealingCallback()

        # Let's record the initial result
        # We will use the context -1 to indicate the initial search
        callback.record_intermediate_data(x0, fn(x0), -1)

        # Get the result
        start_time = time.time()
        result = dual_annealing(
            fn,
            updated_bounds,
            x0=x0,
            maxiter=1000,
            initial_temp=init_temp,
            restart_temp_ratio=res_temp,
            visit=vis,
            accept=acpt,
            callback=callback.record_intermediate_data,
        )

        end_time = time.time()
        total_time = end_time - start_time
        x_tuple = tuple(x for x in result.x)
        fn_values.append(x_tuple + (result.fun, 'Dual Annealing', total_time))

        # Append callback results
        callbacks.append(callback)

    return {'results': results, 'final_results': fn_values, 'callbacks': callbacks}


def run_basinhopping(problem: Dict, trials: int):
    """
    Function that runs basinhopping for a
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
    results = np.zeros((trials, max_iterations * 10, dimensions + 1)) * np.nan
    fn_values = []
    callbacks = []

    for trial in range(trials):
        # Set a random seed
        np.random.seed(trial)

        # Initial point
        x0 = initialize_vector(size=dimensions, bounds=bounds)

        # Set up the function with the results
        fn = lambda x: objective(  # noqa
            x, results=results, trial=trial, version='numpy'
        )

        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}

        # Get the result
        start_time = time.time()
        result = basinhopping(
            fn,
            x0=x0,
            niter=1000,
            minimizer_kwargs=minimizer_kwargs,
        )

        end_time = time.time()
        total_time = end_time - start_time
        x_tuple = tuple(x for x in result.x)
        fn_values.append(x_tuple + (result.fun, 'Basinhopping', total_time))
        callbacks.append(0)

    return {'results': results, 'final_results': fn_values, 'callbacks': callbacks}


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
    results = np.zeros((trials, max_iterations * 10, dimensions + 1)) * np.nan
    fn_values = []
    callbacks = []

    for trial in range(trials):
        # Set a random seed
        np.random.seed(trial)

        # Initial point
        x0 = initialize_vector(size=dimensions, bounds=bounds)

        # Callback
        callback = DifferentialEvolutionCallback()

        # Let's record the initial result
        # We will use the context -1 to indicate the initial search
        callback.record_intermediate_data(x0, -1)

        # Set up the function with the results
        fn = lambda x: objective(  # noqa
            x, results=results, trial=trial, version='numpy'
        )

        # Get the result
        start_time = time.time()
        result = differential_evolution(
            fn,
            updated_bounds,
            x0=x0,
            maxiter=1000,
            strategy=strat,
            mutation=mut,
            recombination=recomb,
            callback=callback.record_intermediate_data,
        )
        end_time = time.time()
        total_time = end_time - start_time
        x_tuple = tuple(x for x in result.x)
        fn_values.append(x_tuple + (result.fun, 'Differential Evolution', total_time))

        # Append callback results
        callbacks.append(callback)

    return {'results': results, 'final_results': fn_values, 'callbacks': callbacks}


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
    x = torch.stack(x_values)

    # Box constraints
    if len(bounds) > 2:
        a, b = bounds[0]
        x_scaled = a + (b - a) / 2.0 * (torch.sin(x) + 1)
    else:
        scaled_x_values = []
        for index, constr in enumerate(bounds):
            a, b = constr
            x_constr = a + (b - a) / 2.0 * (torch.sin(x[index]) + 1)
            scaled_x_values.append(x_constr)
        x_scaled = torch.stack(scaled_x_values)

    f = objective(x_scaled)

    # # Setup the bounds for the inequality
    # # a <= x <= b
    # # -> a - x <= 0
    # # -> x - b <= 0
    # # Inequality constraints
    # # In some cases all of the bounds will be
    # # None so we will need to set the ci struct
    # # to None
    ci = None

    # # TODO: I will leave this for now but it was not working - is there
    # # an implementation issue?
    # if np.any(np.array(bounds).flatten() != None):  # noqa
    #     ci = pygransoStruct()
    #     for index, cnstr in enumerate(bounds):
    #         a, b = cnstr
    #         if a is None and b is None:
    #             pass
    #         elif a is None:
    #             setattr(ci, f'ca{index}', a - x[index])
    #         elif b is None:
    #             setattr(ci, f'cb{index}', x[index] - b)
    #         else:
    #             setattr(ci, f'ca{index}', a - x[index])
    #             setattr(ci, f'cb{index}', x[index] - b)

    # Equality constraints
    ce = None

    return f, ci, ce


def run_pygranso(problem: Dict, trials: int):
    """
    Function that runs PyGranso for our deeplifting
    comparisions
    """
    # Get the device (CPU for now)
    device = torch.device('cpu')
    fn_values = []
    interim_results = []

    # Get the maximum iterations
    max_iterations = problem['max_iterations']

    # Get the number of dimensions for the problem
    dimensions = problem['dimensions']

    # results
    results = np.zeros((trials, max_iterations * 100, dimensions + 1)) * np.nan

    for trial in range(trials):
        # Seed everything
        set_seed(trial)

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
        # Initial point
        x0 = initialize_vector(size=dimensions, bounds=bounds)
        x0 = torch.tensor(x0).reshape(dimensions, 1)
        x0 = x0.to(device=device, dtype=torch.double)

        # Pygranso options
        opts.x0 = x0
        opts.torch_device = device
        # opts.print_frequency = 100
        opts.print_level = 0
        opts.limited_mem_size = 5
        opts.stat_l2_model = False
        opts.double_precision = True
        opts.viol_ineq_tol = 1e-10
        opts.opt_tol = 1e-10
        opts.maxit = 2000

        # Objective function
        objective = problem['objective']

        # Set up the function with the results
        fn = lambda x: objective(  # noqa
            x, results=results, trial=trial, version='pytorch'
        )

        # Combined function
        comb_fn = lambda x: pygranso_nd_fn(x, fn, bounds)  # noqa

        # Initiate halt log
        mHLF_obj = HaltLog()
        halt_log_fn, get_log_fn = mHLF_obj.makeHaltLogFunctions(opts.maxit)

        #  Set PyGRANSO's logging function in opts
        opts.halt_log_fn = halt_log_fn

        # Run the main algorithm
        start_time = time.time()
        soln = pygranso(var_spec=var_in, combined_fn=comb_fn, user_opts=opts)
        end_time = time.time()
        total_time = end_time - start_time

        # GET THE HISTORY OF ITERATES
        # Even if an error is thrown, the log generated until the error can be
        # obtained by calling get_log_fn()
        log = get_log_fn()

        # Final structure
        indexes = (pd.Series(log.fn_evals).cumsum() - 1).values.tolist()

        # Index results
        interim_results.append(results[trial, indexes, :dimensions])

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

    return {'results': None, 'final_results': fn_values, 'callbacks': interim_results}


def deeplifting_predictions(x, objective, method='particle'):
    """
    Convert scaled values to the objective function
    """
    if method == 'particle':
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

    elif method == 'single-value':
        x = x.mean(axis=0)
        f = objective(x)

        # Detach and return x
        x = x.detach().cpu().numpy().flatten()

    else:
        raise ValueError(f'{method} does not exist!')

    return x, f


def deeplifting_nd_fn(
    model,
    objective,
    trial,
    dimensions,
    deeplifting_results,
    problem_name,
    method='particle',
    inputs=None,
):
    """
    Combined funtion used for PyGranso
    """
    outputs = model(inputs=inputs)

    # Get x1 and x2 so we can add the bounds
    # outputs = torch.sigmoid(outputs)
    # x = outputs.mean(axis=0)
    # print(f'Output x {x}')
    x, f = deeplifting_predictions(outputs, objective, method=method)
    # print(x, f)
    # f_copy = f.detach().cpu().numpy()

    # # Fill in the intermediate results
    # if problem_name not in EXCLUDE_PROBLEMS:
    #     iteration = np.argmin(~np.any(np.isnan(deeplifting_results[trial]), axis=1))
    #     deeplifting_results[trial, iteration, :dimensions] = x
    #     deeplifting_results[trial, iteration, -1] = f_copy

    # Inequality constraint
    ci = None

    # Equality constraing
    ce = None

    return f, ci, ce


def run_deeplifting(
    problem: Dict,
    problem_name: str,
    trials: int,
    input_size=512,
    hidden_sizes=(512, 512, 512),
    activation='sine',
    output_activation='sine',
    agg_function='sum',
    include_bn=False,
    method='particle',
    save_model_path=None,
):
    """
    Function that runs our preimer method of deeplifting.
    The idea here is to reparmeterize an optimization objective
    so we can have better optimization and convergence
    """
    # Get the device (CPU for now)
    output_size = problem['dimensions']
    device = get_devices()
    fn_values = []
    iterim_results: List[Any] = []  # noqa

    # Get the initial starting point for each trial
    initial_values = []

    # Keep the termination codes for each of the trials
    termination_codes = []

    for trial in range(trials):
        set_seed(trial)
        # Objective function
        objective = problem['objective']

        # Get the bounds of the problem
        if output_size <= 2:
            bounds = problem['bounds']
        else:
            bounds = problem['bounds']

            if len(bounds) > 1:
                bounds = bounds

            else:
                bounds = bounds * output_size

        # Fix the inputs for deeplifting
        inputs = torch.randn(1, 5 * output_size)
        inputs = inputs.to(device=device, dtype=torch.double)

        x0 = initialize_vector(size=output_size, bounds=bounds)
        x0 = torch.tensor(x0)
        x0 = x0.to(device=device, dtype=torch.double)

        # Deeplifting model with skip connections
        model = DeepliftingSkipMLP(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            bounds=bounds,
            skip_every_n=1,
            activation=activation,
            output_activation=output_activation,
            agg_function=agg_function,
            include_bn=include_bn,
            seed=trial,
        )

        model = model.to(device=device, dtype=torch.double)

        # # Let's make sure that all methods have the same x0
        # train_model_to_output(
        #     inputs=inputs, model=model, x0=x0, epochs=100000, lr=1e-4, tolerance=1e-3
        # )
        nvar = getNvarTorch(model.parameters())

        # Setup a pygransoStruct for the algorithm
        # options
        opts = pygransoStruct()

        # Print the model outputs and check against x0 also
        # want to use a print out to make sure all models have the same starting
        # point
        model.eval()
        outputs = model(inputs=inputs)
        outputs = outputs.mean(axis=0)

        print(f'Initial x0 = {x0}')
        print(f'Fitted x0 = {outputs}')
        initial_values.append(outputs.detach().cpu().numpy())

        # Inital x0
        x0 = (
            torch.nn.utils.parameters_to_vector(model.parameters())
            .detach()
            .reshape(nvar, 1)
            .to(device=device, dtype=torch.double)
        )

        opts.x0 = x0
        opts.torch_device = device
        # opts.print_level = 0
        opts.print_frequency = 1
        opts.limited_mem_size = 100
        opts.stat_l2_model = False
        opts.double_precision = True
        # opts.disable_terminationcode_6 = True
        # opts.halt_on_linesearch_bracket = False
        opts.opt_tol = 1e-10
        opts.maxit = 5000

        # Get the maximum iterations
        max_iterations = problem['max_iterations']  # noqa

        # TODO: Clean up meaningless variables
        results = None
        deeplifting_results = None

        # Set up the function with the results
        fn = lambda x: objective(  # noqa
            x, results=results, trial=trial, version='pytorch'
        )

        # Get the objective value at the initial point
        outputs = model(inputs=inputs)

        init_fn = lambda x: objective(  # noqa
            x, results=None, trial=0, version='pytorch'
        )
        x_init, f_init = deeplifting_predictions(outputs, init_fn, method=method)
        f_init = f_init.detach().cpu().numpy()

        # Combined function
        comb_fn = lambda model: deeplifting_nd_fn(  # noqa
            model,
            fn,
            trial,
            output_size,
            deeplifting_results,
            problem_name=problem_name,
            method=method,
            inputs=inputs,
        )  # noqa

        # Run the main algorithm
        model.train()
        start_time = time.time()
        soln = pygranso(var_spec=model, combined_fn=comb_fn, user_opts=opts)
        end_time = time.time()
        total_time = end_time - start_time

        # Save the termination code
        termination_codes.append(soln.termination_code)

        # Get final x we will also need to map
        # it to the same bounds
        outputs = model(inputs=inputs)

        # Get the final results
        final_results = None
        final_fn = lambda x: objective(  # noqa
            x, results=final_results, trial=0, version='pytorch'
        )
        xf, f = deeplifting_predictions(outputs, final_fn, method=method)

        f = f.detach().cpu().numpy()
        data_point = tuple(xf) + (float(f), float(f_init), 'Deeplifting', total_time)
        fn_values.append(data_point)

        # If save model then we need to save the configuration of the
        # model and the model weights
        if save_model_path is not None:
            # Add success criteria to the file name
            status = np.abs(f - problem['global_minimum']) <= 1e-4

            # Get the configuration
            config = {}
            config['input_size'] = input_size
            config['hidden_sizes'] = hidden_sizes
            config['dimensions'] = output_size
            config['bounds'] = bounds
            config['activation'] = activation
            config['output_activation'] = output_activation
            config['agg_function'] = agg_function
            config['seed'] = trial
            config['global_minimum'] = problem['global_minimum']

            # Save the model
            model_file_name = os.path.join(
                save_model_path, f'{problem_name}-{trial}-{status}.pt'
            )
            torch.save(model.state_dict(), model_file_name)

            # Save model config json
            json_file_name = os.path.join(
                save_model_path, f'config-{problem_name}-{trial}-{status}.json'
            )
            with open(json_file_name, 'w') as json_file:
                json.dump(config, json_file, indent=4)

        # Collect garbage and empty cache
        del (model, nvar, x0, opts, soln, outputs, xf, f, data_point, final_fn)
        gc.collect()
        torch.cuda.empty_cache()

    return {
        'results': None,
        'final_results': fn_values,
        'callbacks': iterim_results,
        'initial_values': initial_values,
        'termination_codes': termination_codes,
    }


def deeplifting_high_dimension_fn(model, objective, inputs=None):
    """
    Combined funtion used for PyGranso
    """
    outputs = model(inputs=inputs)
    outputs = outputs.mean(axis=0)
    f = objective(outputs, version='pytorch')

    # Inequality constraint
    ci = None

    # Equality constraing
    ce = None

    return f, ci, ce


def run_high_dimensional_deeplifting(
    problem: Dict,
    problem_name: str,
    trials: int,
    input_size=512,
    hidden_sizes=(512, 512, 512),
    activation='sine',
    output_activation='sine',
    agg_function='sum',
    include_bn=False,
):
    """
    Function that runs our preimer method of deeplifting.
    The idea here is to reparmeterize an optimization objective
    so we can have better optimization and convergence
    """
    # Get the device (CPU for now)
    output_size = problem['dimensions']
    device = get_devices()
    fn_values = []

    for trial in range(trials):
        set_seed(trial)
        # Objective function
        objective = problem['objective']

        # Get the bounds of the problem
        if output_size <= 2:
            bounds = problem['bounds']
        else:
            bounds = problem['bounds']

            if len(bounds) > 1:
                bounds = bounds

            else:
                bounds = bounds * output_size

        # Fix the inputs for deeplifting
        inputs = torch.randn(1, 5 * output_size)
        inputs = inputs.to(device=device, dtype=torch.double)

        x0 = initialize_vector(size=output_size, bounds=bounds)
        x0 = torch.tensor(x0)
        x0 = x0.to(device=device, dtype=torch.double)

        # Deeplifting model with skip connections
        model = DeepliftingSkipMLP(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            bounds=bounds,
            skip_every_n=1,
            activation=activation,
            output_activation=output_activation,
            agg_function=agg_function,
            include_bn=include_bn,
            seed=trial,
        )

        model = model.to(device=device, dtype=torch.double)

        # Let's make sure that all methods have the same x0
        print('Set weights to match x0')
        train_model_to_output(
            inputs=inputs, model=model, x0=x0, epochs=100000, lr=1e-4, tolerance=1e-10
        )

        # Let's also get the initial values of the objective
        # so we can normalize the results
        model.eval()
        outputs = model(inputs=inputs)
        outputs = outputs.mean(axis=0)
        f_init = objective(outputs, version='pytorch')

        # put model back in training mode
        model.train()
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
        opts.print_level = 0
        # opts.print_frequency = 1
        opts.limited_mem_size = 100
        opts.stat_l2_model = False
        opts.double_precision = True
        opts.disable_terminationcode_6 = True
        opts.halt_on_linesearch_bracket = False
        opts.linesearch_maxit = 50
        opts.opt_tol = 1e-10
        opts.maxit = 5

        # # Combined function
        comb_fn = lambda model: deeplifting_high_dimension_fn(
            model, objective, inputs
        )  # noqa

        # Run the main algorithm
        start_time = time.time()
        soln = pygranso(var_spec=model, combined_fn=comb_fn, user_opts=opts)
        end_time = time.time()
        total_time = end_time - start_time

        # Get final x we will also need to map
        # it to the same bounds
        outputs = model(inputs=inputs)
        outputs = outputs.mean(axis=0)
        final_fn = objective(outputs, version='pytorch')
        f = final_fn.detach().cpu().numpy()
        xf = outputs.detach().cpu().numpy()
        print(f'Final value = {f}')

        data_point = tuple(xf) + (float(f), float(f_init), 'Deeplifting', total_time)
        fn_values.append(data_point)

        # Collect garbage and empty cache
        del (model, nvar, x0, opts, soln, outputs, xf, f, data_point, final_fn)
        gc.collect()
        torch.cuda.empty_cache()

    return {'results': None, 'final_results': fn_values, 'callbacks': []}


def run_lbfgs_deeplifting(
    problem: Dict,
    problem_name: str,
    trials: int,
    input_size=512,
    hidden_sizes=(512, 512, 512),
    activation='sine',
    output_activation='sine',
    agg_function='sum',
    include_bn=False,
):
    """
    Function that runs our preimer method of deeplifting.
    The idea here is to reparmeterize an optimization objective
    so we can have better optimization and convergence. The difference
    with this version of deeplifting is we will use
    pytorch's built in LBFGS algorithm
    """
    # Get the device (CPU for now)
    output_size = problem['dimensions']
    device = get_devices()
    fn_values = []  # noqa

    for trial in range(trials):
        set_seed(trial)
        # Objective function
        objective = problem['objective']

        # Get the bounds of the problem
        if output_size <= 2:
            bounds = problem['bounds']
        else:
            bounds = problem['bounds']

            if len(bounds) > 1:
                bounds = bounds

            else:
                bounds = bounds * output_size

        # Fix the inputs for deeplifting
        inputs = torch.randn(1, 5 * output_size)
        inputs = inputs.to(device=device, dtype=torch.double)

        x0 = initialize_vector(size=output_size, bounds=bounds)
        x0 = torch.tensor(x0)
        x0 = x0.to(device=device, dtype=torch.double)

        # Deeplifting model with skip connections
        model = DeepliftingSkipMLP(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            bounds=bounds,
            skip_every_n=1,
            activation=activation,
            output_activation=output_activation,
            agg_function=agg_function,
            include_bn=include_bn,
            seed=trial,
        )

        model = model.to(device=device, dtype=torch.double)

        # Let's make sure that all methods have the same x0
        print('Set weights to match x0')
        train_model_to_output(
            inputs=inputs, model=model, x0=x0, epochs=100000, lr=1e-4, tolerance=1e-3
        )

        # Set up the optimizer for the problem
        optimizer = optim.LBFGS(
            model.parameters(),
            lr=1.0,
            history_size=200,
            max_iter=50,
            line_search_fn='strong_wolfe',
        )

        # Set up a training loop
        start = time.time()

        # Get starting loss
        outputs = model(inputs=inputs)
        outputs = outputs.mean(axis=0)
        f_init = current_loss = objective(outputs, version='pytorch')
        print(f'Initial loss = {current_loss}')

        count = 0
        for epoch in range(100000):

            def closure():
                # Zero out the gradients
                optimizer.zero_grad()

                # The loss is the sum of the compliance
                outputs = model(inputs=inputs)
                outputs = outputs.mean(axis=0)
                loss = objective(outputs, version='pytorch')

                # Go through the backward pass and create the gradients
                loss.backward()

                return loss

            # Step through the optimzer to update the data with the gradients
            optimizer.step(closure)

            outputs = model(inputs=inputs)
            outputs = outputs.mean(axis=0)
            updated_loss = objective(outputs, version='pytorch')

            # Break loop if tolerance is met
            flat_grad = optimizer._gather_flat_grad()  # type: ignore
            opt_cond = flat_grad.abs().max() <= 1e-10
            if opt_cond:
                break

            if torch.abs(updated_loss - current_loss) <= 1e-10:
                count += 1
            else:
                current_loss = updated_loss

            if count > 10000:
                break

            if epoch % 1000 == 0:
                print(
                    f'loss = {updated_loss.detach()},'
                    f'gradient_norm = {flat_grad.abs().max()}'
                )

        end = time.time()
        total_time = end - start

        # Get final x we will also need to map
        # it to the same bounds
        outputs = model(inputs=inputs)
        outputs = outputs.mean(axis=0)
        final_fn = objective(outputs, version='pytorch')
        f = final_fn.detach().cpu().numpy()
        xf = outputs.detach().cpu().numpy()
        data_point = tuple(xf.flatten()) + (
            float(f),
            float(f_init),
            'Deeplifting-LBFGS',
            total_time,
        )
        fn_values.append(data_point)

        # Collect garbage and empty cache
        del (outputs, xf, f, data_point, final_fn)
        gc.collect()
        torch.cuda.empty_cache()

    return {'results': None, 'final_results': fn_values, 'callbacks': []}


def run_adam_deeplifting(
    problem: Dict,
    problem_name: str,
    trials: int,
    input_size=512,
    hidden_sizes=(512, 512, 512),
    activation='sine',
    output_activation='sine',
    agg_function='sum',
    include_bn=False,
):
    """
    Function that runs our preimer method of deeplifting.
    The idea here is to reparmeterize an optimization objective
    so we can have better optimization and convergence. The difference
    with this version of deeplifting is we will use
    pytorch's built in LBFGS algorithm
    """
    # Get the device (CPU for now)
    output_size = problem['dimensions']
    device = get_devices()

    # Final values
    fn_values = []

    for trial in range(trials):
        set_seed(trial)
        # Objective function
        objective = problem['objective']

        # Get the bounds of the problem
        if output_size <= 2:
            bounds = problem['bounds']
        else:
            bounds = problem['bounds']

            if len(bounds) > 1:
                bounds = bounds

            else:
                bounds = bounds * output_size

        # Fix the inputs for deeplifting
        inputs = torch.randn(1, 5 * output_size)
        inputs = inputs.to(device=device, dtype=torch.double)

        x0 = initialize_vector(size=output_size, bounds=bounds)
        x0 = torch.tensor(x0)
        x0 = x0.to(device=device, dtype=torch.double)

        # Deeplifting model with skip connections
        model = DeepliftingSkipMLP(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            bounds=bounds,
            skip_every_n=1,
            activation=activation,
            output_activation=output_activation,
            agg_function=agg_function,
            include_bn=include_bn,
            seed=trial,
        )

        model = model.to(device=device, dtype=torch.double)

        # Let's make sure that all methods have the same x0
        print('Set weights to match x0')
        train_model_to_output(
            inputs=inputs, model=model, x0=x0, epochs=100000, lr=1e-4, tolerance=1e-3
        )

        # Set up the optimizer for the problem
        epochs = 10000
        optimizer = optim.Adam(
            model.parameters(),
            lr=1e-4,
        )
        scheduler = OneCycleLR(
            optimizer,
            max_lr=1e-4,
            epochs=epochs,
            steps_per_epoch=1,
            pct_start=0.0,
        )

        # Set up a training loop
        start = time.time()

        # Get starting loss
        outputs = model(inputs=inputs)
        outputs = outputs.mean(axis=0)
        f_init = objective(outputs, version='pytorch')

        # Check
        print('Check that outputs match ... \n')
        print(torch.norm(outputs - x0, p=2).item())

        early_stopping_count = 0
        for epoch in range(epochs):
            # Zero out the gradients
            optimizer.zero_grad()

            # The loss is the sum of the compliance
            outputs = model(inputs=inputs)
            outputs = outputs.mean(axis=0)
            loss = objective(outputs, version='pytorch')

            # Go through the backward pass and create the gradients
            loss.backward()

            outputs = model(inputs=inputs)
            outputs = outputs.mean(axis=0)
            updated_loss = objective(outputs, version='pytorch')

            if epoch % 1000 == 0:
                print(f'loss = {updated_loss.detach()},')

            delta = f_init.item() - updated_loss.item()
            if epoch >= 10000 and delta < 1e-10:
                early_stopping_count += 1

            if early_stopping_count == 10000:
                break

            # Step through the optimzer to update the data with the gradients
            optimizer.step()
            scheduler.step()

        # Get the final results
        end = time.time()
        total_time = end - start  # noqa

        # Get final x we will also need to map
        # it to the same bounds
        outputs = model(inputs=inputs)
        outputs = outputs.mean(axis=0)
        final_fn = objective(outputs, version='pytorch')
        f = final_fn.detach().cpu().numpy()
        xf = outputs.detach().cpu().numpy()

        # Make sure f_init is detached from graph
        f_init = f_init.detach().cpu().numpy()
        data_point = tuple(xf.flatten()) + (
            float(f),
            float(f_init),
            'Deeplifting-Adam',
            total_time,
        )
        fn_values.append(data_point)

        # Collect garbage and empty cache
        gc.collect()
        torch.cuda.empty_cache()

    return {'results': None, 'final_results': fn_values, 'callbacks': []}


def run_pyomo(problem, trials, method):
    """
    Function to run the pyomo module. We will use this function
    to execute optimizations methods BARON and SCIP. Pyomo can
    also handle IPOPT.
    """
    # Check methods
    if method not in ('baron', 'scip'):
        raise ValueError(f'Method {method} is not supported!')

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

    # Initial time limit
    # If the initial time limit is exceeded
    # then we will reduce it within the loop
    time_limit = 60 * 60 * 8

    for trial in range(trials):
        print(f'Pyomo trial = {trial + 1}')
        # Set the random seed
        set_seed(trial)

        # Initial guess (starting point for PYOMO methods)
        # TODO: Need to provide a better starting point here
        x0 = initialize_vector(size=dimensions, bounds=bounds)

        # Get the objective
        fn = lambda x: objective(  # noqa
            x, results=results, trial=trial, version='pyomo'
        )  # noqa

        start_time = time.time()

        # Create the Pyomo model
        model = ConcreteModel()

        # Define the variables with initialization
        init_dict = {i: x0[i] for i in range(dimensions)}
        model.x = Var(range(dimensions), initialize=init_dict)

        # Need to set the bounds here
        for index, bound in enumerate(bounds):
            a, b = bound
            model.x[index].setlb(a)
            model.x[index].setub(b)

        # Create an objective rule
        def objective_rule(model):
            x = np.array([model.x[i] for i in range(dimensions)])
            return fn(x)

        # Set up model objective and we want to minimize expressions
        model.obj = Objective(rule=objective_rule, sense=minimize)

        # Solve the model
        solver = SolverFactory(method)

        # Set an 8 hour time limit
        solver.options['limits/time'] = time_limit
        solver_result = solver.solve(model)
        fn_objective = model.obj()

        # Message
        termination_condition = solver_result['Solver'][0]['Message']
        if termination_condition == 'time limit reached':
            # If the first trial takes 8 hours then reset
            # the limit to 1 minute to get through the rest
            # of the trials
            time_limit = 60

        end_time = time.time()
        total_time = end_time - start_time

        x_tuple = tuple(model.x.get_values().values())
        fn_values.append(x_tuple + (fn_objective, method.upper(), total_time))

    return {'results': results, 'final_results': fn_values}
