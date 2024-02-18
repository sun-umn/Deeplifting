# stdlib
import gc
import json
import time
from typing import Any, Callable, Dict

# third party
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from pygranso.private.getNvar import getNvarTorch
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from pyomo.environ import ConcreteModel, Objective, SolverFactory, Var, minimize
from scipy.optimize import basinhopping, differential_evolution, dual_annealing

# first party
from deeplifting.models import ReLUDeepliftingMLP
from deeplifting.problems import HIGH_DIMENSIONAL_PROBLEMS_BY_NAME
from deeplifting.utils import (
    BasinHoppingCallback,
    DifferentialEvolutionCallback,
    DualAnnealingCallback,
    HaltLog,
    PyGransoConfig,
    initialize_vector,
    set_seed,
    train_model_to_output,
)

# # third party
# import cyipopt
# import jax.numpy as jnp
# from jax import grad, jit


# Do NOT save intermediate paths for high dimensional problems
EXCLUDE_PROBLEMS = list(HIGH_DIMENSIONAL_PROBLEMS_BY_NAME.keys())


# def run_ipopt(problem: Dict, trials: int) -> pd.DataFrame:
#     """
#     Function that runs IPOPT on one of our test
#     functions for deeplifting.
#     """
#     # Objective function
#     objective = problem['objective']

#     # Get the number of dimensions
#     dimensions = problem['dimensions']

#     # Get the bounds of the problem
#     bounds = problem['bounds']
#     upper_bounds = bounds['upper_bounds']
#     lower_bounds = bounds['lower_bounds']
#     list_bounds = list(zip(lower_bounds, upper_bounds))

#     # NOTE: With this setup we can extract the intermediate
#     # values for analysis
#     class IPOPTProblem:
#         def __init__(self, objective_fn):
#             self.objective_fn = objective_fn
#             self.iterations = []
#             self.f_history = []
#             self.nfev = 0

#         def objective(self, x):
#             self.nfev += 1
#             return self.objective_fn(x)

#         def gradient(self, x):
#             obj_jit = jit(self.objective_fn)
#             gradient = grad(obj_jit)
#             return gradient(x)

#         def constraints(self, x):
#             return jnp.zeros_like(x)

#         def intermediate(
#             self,
#             alg_mod,
#             iter_count,
#             obj_value,
#             inf_pr,
#             inf_du,
#             mu,
#             d_norm,
#             regularization_size,
#             alpha_du,
#             alpha_pr,
#             ls_trials,
#         ):
#             """
#             Function to save the history of the objective function values during
#             iteration and also save the number of iterations
#             """
#             self.f_history.append(obj_value)
#             self.iterations.append(iter_count)

#     # Save the results
#     # We will store the optimization steps here
#     trial_results = []
#     for trial in range(trials):
#         # Set the random seed
#         set_seed(trial)

#         # Initial guess (starting point for IPOPT)
#         # TODO: Need to provide a better starting point here
#         x0 = initialize_vector(size=dimensions, bounds=bounds)

#         columns = [f'x{i + 1}' for i in range(dimensions)]
#         xs = json.dumps(dict(zip(columns, x0)))

#         # Get the objective
#         fn = lambda x: objective(x, version='jax')  # noqa  # noqa

#         # Get the initial objective galue
#         f_init = fn(x0)

#         # IPOPT Problem
#         ipopt_problem = IPOPTProblem(objective_fn=fn)
#         nlp = cyipopt.Problem(
#             n=len(x0),
#             m=0,
#             problem_obj=ipopt_problem,
#             lb=list_bounds[0],
#             ub=list_bounds[1],
#         )

#         # Call IPOPT
#         start_time = time.time()

#         # Use the solve method
#         x, info = nlp.solve(x0)

#         end_time = time.time()
#         total_time = end_time - start_time

#         results = {
#             'xs': xs,
#             'f_init': float(f_init),
#             'total_time': total_time,
#             'f_final': info['obj_val'],
#             'iterations': len(ipopt_problem.iterations),
#             'fn_evals': ipopt_problem.nfev,
#             'termination_code': info['status'],
#             'objective_values': np.array(ipopt_problem.f_history),
#         }
#         trial_results.append(results)

#     return pd.DataFrame(trial_results)


def run_dual_annealing(
    problem: Dict,
    trials: int,
    maxiter: int = 1000,
    init_temp: int = 5230,
    res_temp: float = 2e-5,
    vis: float = 2.62,
    acpt: float = -5.0,
) -> pd.DataFrame:
    """
    Function that runs dual annealing for a
    specified optimization problem
    """
    # Objective function
    objective = problem['objective']

    # Get the number of dimensions
    dimensions = problem['dimensions']

    # Get the bounds of the problem
    bounds = problem['bounds']
    upper_bounds = bounds['upper_bounds']
    lower_bounds = bounds['lower_bounds']
    list_bounds = list(zip(lower_bounds, upper_bounds))

    # Run the trials and save the results
    trial_results = []
    for trial in range(trials):
        # Set a random seed
        np.random.seed(trial)

        # Initial point
        x0 = initialize_vector(size=dimensions, bounds=bounds)

        columns = [f'x{i + 1}' for i in range(dimensions)]
        xs = json.dumps(dict(zip(columns, x0)))

        # Set up the function with the results
        fn = lambda x: objective(x, version='numpy')  # noqa

        # Get the initial objective galue
        f_init = fn(x0)

        # Callback
        callback = DualAnnealingCallback()

        # Let's record the initial result
        # We will use the context -1 to indicate the initial search
        callback.record_intermediate_data(x0, fn(x0), -1)

        # Get the result
        start_time = time.time()
        result = dual_annealing(
            fn,
            list_bounds,
            x0=x0,
            maxiter=maxiter,
            initial_temp=init_temp,
            restart_temp_ratio=res_temp,
            visit=vis,
            accept=acpt,
            callback=callback.record_intermediate_data,
        )

        end_time = time.time()
        total_time = end_time - start_time

        results = {
            'xs': xs,
            'initial_temp': init_temp,
            'f_init': f_init,
            'total_time': total_time,
            'f_final': result.fun,
            'iterations': result.nit,
            'fn_evals': result.nfev,
            'termination_code': result.status,
            'objective_values': np.array(callback.f_history),
        }
        trial_results.append(results)

    return pd.DataFrame(trial_results)


def run_basinhopping(problem: Dict, trials: int, niter: int, T: float) -> pd.DataFrame:
    """
    Function that runs basinhopping for a
    specified optimization problem
    """
    # Objective function
    objective = problem['objective']

    # Get the number of dimensions
    dimensions = problem['dimensions']

    # Get the bounds of the problem
    bounds = problem['bounds']
    upper_bounds = bounds['upper_bounds']
    lower_bounds = bounds['lower_bounds']
    list_bounds = list(zip(lower_bounds, upper_bounds))

    # Run the trials and save the results
    trial_results = []
    for trial in range(trials):
        # Set a random seed
        np.random.seed(trial)

        # Initial point
        x0 = initialize_vector(size=dimensions, bounds=bounds)

        columns = [f'x{i + 1}' for i in range(dimensions)]
        xs = json.dumps(dict(zip(columns, x0)))

        # Set up the function with the results
        fn = lambda x: objective(x, version='numpy')  # noqa

        # Get the initial objective galue
        f_init = fn(x0)

        # Callback
        callback = BasinHoppingCallback()

        # Let's record the initial result
        # We will use the context -1 to indicate the initial search
        callback.record_intermediate_data(x0, fn(x0), -1)

        # Get the result
        start_time = time.time()

        # NOTE: Bounds uses minimizer kwards
        minimizer_kwargs = {'bounds': list_bounds}

        result = basinhopping(
            fn,
            x0=x0,
            niter=niter,
            T=T,
            minimizer_kwargs=minimizer_kwargs,
            callback=callback.record_intermediate_data,
        )

        end_time = time.time()
        total_time = end_time - start_time

        results = {
            'xs': xs,
            'T': T,
            'f_init': f_init,
            'total_time': total_time,
            'f_final': result.fun,
            'iterations': result.nit,
            'fn_evals': result.nfev,
            'termination_code': result.lowest_optimization_result.status,
            'objective_values': np.array(callback.f_history),
        }
        trial_results.append(results)

    return pd.DataFrame(trial_results)


def run_differential_evolution(
    problem: Dict,
    trials: int,
    maxiter: int = 1000,
    popsize: int = 15,
    mutation: float = 0.5,
    recombination: float = 0.7,
) -> pd.DataFrame:
    """
    Function that runs differential evolution for a
    specified optimization problem

    The documentations states that in order to increase the likelihood
    of finding a global minimum use:
    1. high popsize values
    2. high mutation
    3. lower recombination
    """
    # Objective function
    objective = problem['objective']

    # Get the number of dimensions
    dimensions = problem['dimensions']

    # Get the bounds of the problem
    bounds = problem['bounds']
    upper_bounds = bounds['upper_bounds']
    lower_bounds = bounds['lower_bounds']
    list_bounds = list(zip(lower_bounds, upper_bounds))

    # Iterate over the different starting positions
    trial_results = []
    for trial in range(trials):
        # Set a random seed
        np.random.seed(trial)

        # Initial point
        x0 = initialize_vector(size=dimensions, bounds=bounds)

        columns = [f'x{i + 1}' for i in range(dimensions)]
        xs = json.dumps(dict(zip(columns, x0)))

        # Set up the function with the results
        fn = lambda x: objective(x, version='numpy')  # noqa

        # Get the initial objective galue
        f_init = fn(x0)

        # Callback
        callback = DifferentialEvolutionCallback()

        # Get the result
        start_time = time.time()
        result = differential_evolution(
            fn,
            list_bounds,
            x0=x0,
            maxiter=maxiter,
            popsize=popsize,
            mutation=mutation,
            recombination=recombination,
            callback=callback.record_intermediate_data,
            workers=1,  # Important because we want to turn off multiprocessing
        )

        end_time = time.time()
        total_time = end_time - start_time

        # NOTE: For differential evolution there is no status returned
        # only success = True / False. So I will turn this into the status
        # by turning it into an int
        results = {
            'xs': xs,
            'popsize': popsize,
            'mutation': mutation,
            'recombination': recombination,
            'f_init': f_init,
            'total_time': total_time,
            'f_final': result.fun,
            'iterations': result.nit,
            'fn_evals': result.nfev,
            'termination_code': int(result.success),
            'objective_values': np.array(callback.f_history),
        }
        trial_results.append(results)

    return pd.DataFrame(trial_results)


def pygranso_nd_fn(X_struct, objective, bounds):
    """
    Function to run PyGranso without a neural
    network approximation to the inputs
    """
    # list comprehensions more performant?
    x_values = [value for key, value in X_struct.__dict__.items()]

    # Need to iterate over the struct to get the values
    x = torch.stack(x_values).flatten()

    # Get the bounds from the dictionary
    a = torch.tensor(bounds['lower_bounds'])
    b = torch.tensor(bounds['upper_bounds'])

    # Get the scaled x-values
    x = torch.tanh(x)
    x = a + ((b - a) / 2.0 * (x + 1))

    f = objective(x)

    # Inequality constraints
    ci = None

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

    # Objective function
    objective = problem['objective']

    # Get the number of dimensions
    dimensions = problem['dimensions']

    # Get the bounds of the problem
    bounds = problem['bounds']

    trial_results = []
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

        # Create x0 based on the bounds
        # Initial point
        x0 = initialize_vector(size=dimensions, bounds=bounds)

        # Create the initial xs before we turn into a torch
        # tensor
        columns = [f'x{i + 1}' for i in range(dimensions)]
        xs = json.dumps(dict(zip(columns, x0)))

        # Convert x0 into a torch tensor
        x0 = torch.tensor(x0).reshape(dimensions, 1)
        x0 = x0.to(device=device, dtype=torch.double)

        # Pygranso options
        opts.x0 = x0
        opts.torch_device = device
        opts.print_level = 0
        opts.limited_mem_size = 5
        opts.stat_l2_model = False
        opts.double_precision = True
        opts.viol_ineq_tol = 1e-10
        opts.opt_tol = 1e-10
        opts.maxit = 2000

        # Set up the function with the results
        fn = lambda x: objective(x, version='pytorch')  # noqa

        # Get the initial objective value
        f_init = fn(x0)
        f_init = float(f_init)

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

        # Get the f history
        f_history = np.array(log.f)

        # Get final x we will also need to map
        # it to the same bounds
        x = soln.final.x.cpu().numpy().flatten()

        # Final objective function value
        f = soln.final.f

        results = {
            'xs': xs,
            'f_init': f_init,
            'total_time': total_time,
            'f_final': f,
            'iterations': soln.iters + 1,
            'fn_evals': soln.fn_evals,
            'termination_code': soln.termination_code,
            'objective_values': f_history,
        }
        trial_results.append(results)

        # Delete objects to free memory
        del (var_in, x0, opts, soln, x, f)
        gc.collect()
        torch.cuda.empty_cache()

    return pd.DataFrame(trial_results)


def deeplifting_predictions(x, objective):
    """
    Convert scaled values to the objective function
    """
    f = objective(x)

    # Detach and return x
    x = x.detach().cpu().numpy().flatten()

    return x, f


def deeplifting_nd_fn(
    model,
    objective,
    inputs,
):
    """
    Combined funtion used for PyGranso
    """
    outputs = model(inputs=inputs)

    # Get the predictions and x-values
    x, f = deeplifting_predictions(outputs, objective)

    # Inequality constraint
    ci = None

    # Equality constraing
    ce = None

    return f, ci, ce


def run_pygranso_deeplifting(
    model: ReLUDeepliftingMLP,
    model_inputs: torch.Tensor,
    start_position: torch.Tensor,
    objective: Callable,
    device: torch.device,
    max_iterations: int,
) -> Dict[str, Any]:
    """
    Function that runs the pygranso version of deeplifting.
    """
    # # The first thing we do when training this model is align the
    # # outputs of the model with a randomly initialized starting position
    train_model_to_output(
        inputs=model_inputs,
        model=model,
        x0=start_position,
        tolerance=1e-3,
    )

    # Get the number of training variables for the model
    nvar = getNvarTorch(model.parameters())

    # Get the starting parameters for the neural network
    x0 = (
        torch.nn.utils.parameters_to_vector(model.parameters())
        .detach()
        .reshape(nvar, 1)
        .to(device=device, dtype=torch.double)
    )

    # Gather the starting information for the problem
    model.eval()
    outputs = model(inputs=model_inputs)
    _, f_init = deeplifting_predictions(outputs, objective)
    f_init = f_init.detach().cpu().numpy()

    # Build the combination function - a paradigm specific to
    # PyGranso
    comb_fn = lambda model: deeplifting_nd_fn(  # noqa
        model=model,
        objective=objective,
        inputs=model_inputs,
    )  # noqa

    # Run the main algorithm
    # Set up the PyGranso configuation
    pygranso_config = PyGransoConfig(
        device=device,
        x0=x0,
        max_iterations=max_iterations,
    )

    # Put the model back in training mode
    model.train()

    # Start the timer
    start = time.time()

    # Initiate halt log
    mHLF_obj = HaltLog()
    halt_log_fn, get_log_fn = mHLF_obj.makeHaltLogFunctions(max_iterations)

    #  Set PyGRANSO's logging function in opts
    pygranso_config.opts.halt_log_fn = halt_log_fn

    # Run PyGranso
    soln = pygranso(var_spec=model, combined_fn=comb_fn, user_opts=pygranso_config.opts)

    # GET THE HISTORY OF ITERATES
    # Even if an error is thrown, the log generated until the error can be
    # obtained by calling get_log_fn()
    log = get_log_fn()

    # Get the objective values
    objective_values = np.array(log.f)

    # Get the total time
    end = time.time()
    total_time = end - start

    # We will run this iteratively. Every run of this function will
    # be responsible for a single record towards analysis
    results = {
        'soln': soln,
        'f_init': f_init,
        'total_time': total_time,
        'f_final': soln.best.f,
        'iterations': soln.iters,
        'fn_evals': soln.fn_evals,
        'termination_code': soln.termination_code,
        'objective_values': objective_values,
    }

    return results


def run_lbfgs_deeplifting(
    model: ReLUDeepliftingMLP,
    model_inputs: torch.Tensor,
    start_position: torch.Tensor,
    objective: Callable,
    device: torch.device,
    max_iterations: int,
) -> Dict[str, Any]:
    """
    Function that runs our preimer method of deeplifting.
    The idea here is to reparmeterize an optimization objective
    so we can have better optimization and convergence. The difference
    with this version of deeplifting is we will use
    pytorch's built in LBFGS algorithm
    """
    # The first thing we do when training this model is align the
    # outputs of the model with a randomly initialized starting position
    train_model_to_output(
        inputs=model_inputs,
        model=model,
        x0=start_position,
        tolerance=1e-3,
    )

    # Set up the LBFGS optimizer for the problem
    # This is a pytorch provided implementation
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=0.1,
        history_size=25,
        max_iter=25,
        line_search_fn='strong_wolfe',
    )

    # Let's keep the same steps as the other method defined
    # (pygranso deeplifting)

    # Gather the starting information for the problem
    model.eval()
    outputs = model(inputs=model_inputs)
    _, f_init = deeplifting_predictions(outputs, objective)

    # Start clocking on the training loop
    start = time.time()

    # Count for early exit
    count = 0

    # Total iterations
    iterations = 0

    # Create terminations codes like pygranso
    termination_code = None

    # Set current loss as well
    current_loss = f_init

    # Train the model
    epochs = 1000
    model.train()
    for epoch in range(epochs):

        def closure():
            # Zero out the gradients
            optimizer.zero_grad()

            # The loss is the sum of the compliance
            outputs = model(inputs=model_inputs)
            loss = objective(outputs)

            # Go through the backward pass and create the gradients
            loss.backward()

            return loss

        # Step through the optimzer to update the data with the gradients
        optimizer.step(closure)

        outputs = model(inputs=model_inputs)
        updated_loss = objective(outputs)

        # Update the iteration count
        iterations += 1

        # Break loop if tolerance is met
        # Under this condition we check the L_inf norm
        flat_grad = optimizer._gather_flat_grad()  # type: ignore
        opt_cond = flat_grad.abs().max() <= 1e-10  # Same tolerance as pygranso
        if opt_cond:
            termination_code = 0
            break

        # If the difference between the current objective values
        # and the previous do no change over a period then exit
        if (
            np.abs(
                updated_loss.detach().cpu().numpy()
                - current_loss.detach().cpu().numpy()
            )
            <= 1e-10
        ):
            count += 1
        else:
            # If this is not true we need to restart the count
            count = 0
            current_loss = updated_loss

        if count > 100:
            termination_code = 0
            break

        if epoch % 50 == 0:
            print(
                f'loss = {updated_loss.detach()},'
                f' gradient_norm = {flat_grad.abs().max()}'
            )

    # Maximum iterations reached
    if iterations >= len(range(epochs)):
        termination_code = 4

    # Total time to run algorithm
    end = time.time()
    total_time = end - start

    # Get final x we will also need to map
    # it to the same bounds
    model.eval()
    outputs = model(inputs=model_inputs)

    # Get the final objective value
    f_final = objective(outputs)
    f_final = f_final.detach().cpu().numpy()

    # Get total function evaluations
    fn_evals = list(optimizer.state.items())[0][1]['func_evals']

    results = {
        'f_init': f_init.detach().cpu().numpy(),
        'total_time': total_time,
        'f_final': f_final,
        'iterations': iterations,
        'fn_evals': fn_evals,
        'termination_code': termination_code,
    }

    return results


def run_adam_deeplifting(
    model: ReLUDeepliftingMLP,
    model_inputs: torch.Tensor,
    start_position: torch.Tensor,
    objective: Callable,
    device: torch.device,
    max_iterations: int = 1000,
    *,
    lr: float = 1e-2,
) -> Dict[str, Any]:
    """
    Function that runs our preimer method of deeplifting.
    The idea here is to reparmeterize an optimization objective
    so we can have better optimization and convergence. The difference
    with this version of deeplifting is we will use
    pytorch's built in Adam optimizer
    """
    # The first thing we do when training this model is align the
    # outputs of the model with a randomly initialized starting position
    train_model_to_output(
        inputs=model_inputs,
        model=model,
        x0=start_position,
        tolerance=1e-3,
    )

    # Total epochs
    epochs = max_iterations

    # Set up the LBFGS optimizer for the problem
    # This is a pytorch provided implementation
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        amsgrad=True,
    )

    # Reduce learning rate on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.9,
        patience=1,
        verbose=False,
    )

    # Let's keep the same steps as the other method defined
    # (pygranso deeplifting)

    # Gather the starting information for the problem
    model.eval()
    outputs = model(inputs=model_inputs)
    _, f_init = deeplifting_predictions(outputs, objective)

    # Start clocking on the training loop
    start = time.time()

    # Total iterations
    iterations = 0

    # Create terminations codes like pygranso
    termination_code = None

    # Set up a training loop
    start = time.time()

    # Let's try using the minimum loss found during the
    # optimization
    losses = []

    # Train the model
    model.train()
    for epoch in range(epochs):
        # Zero out the gradients
        optimizer.zero_grad()

        # The loss is the sum of the compliance
        outputs = model(inputs=model_inputs)
        loss = objective(outputs)

        # Go through the backward pass and create the gradients
        loss.backward()
        losses.append(loss.item())

        # Step through the optimzer to update the data with the gradients
        optimizer.step()
        scheduler.step(loss)

        iterations += 1

    # Maximum iterations reached
    if iterations >= len(range(epochs)):
        termination_code = 4

    # Get the final results
    end = time.time()
    total_time = end - start  # noqa

    # Get final values
    outputs = model(inputs=model_inputs)
    f_final = objective(outputs)  # noqa
    f_final = f_final.detach().cpu().numpy()  # noqa

    # F-min
    f_min = np.min(np.array(losses))  # noqa

    results = {
        'f_init': f_init.detach().cpu().numpy(),
        'total_time': total_time,
        'f_final': f_final,
        'iterations': iterations,
        'fn_evals': None,  # Does not apply to this method
        'termination_code': termination_code,
        'objective_values': np.array(losses),
    }

    return results


def run_sgd_deeplifting(
    model: ReLUDeepliftingMLP,
    model_inputs: torch.Tensor,
    start_position: torch.Tensor,
    objective: Callable,
    device: torch.device,
    max_iterations: int = 3000,
    *,
    lr: float = 1e-2,
    momentum: float = 0.99,
) -> Dict[str, Any]:
    """
    Function that runs our preimer method of deeplifting.
    The idea here is to reparmeterize an optimization objective
    so we can have better optimization and convergence. The difference
    with this version of deeplifting is we will use
    pytorch's built in Adam optimizer
    """
    # The first thing we do when training this model is align the
    # outputs of the model with a randomly initialized starting position
    # For now we will not worry about the alignment
    train_model_to_output(
        inputs=model_inputs,
        model=model,
        x0=start_position,
        tolerance=1e-3,
    )

    # Total epochs
    epochs = max_iterations

    # Set up the LBFGS optimizer for the problem
    # This is a pytorch provided implementation
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Reduce learning rate on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.9,
        patience=1,
        verbose=False,
    )

    # Let's keep the same steps as the other method defined
    # (pygranso deeplifting)

    # Gather the starting information for the problem
    model.eval()
    outputs = model(inputs=model_inputs)
    _, f_init = deeplifting_predictions(outputs, objective)

    # Start clocking on the training loop
    start = time.time()

    # Total iterations
    iterations = 0

    # Create terminations codes like pygranso
    termination_code = None

    # Set up a training loop
    start = time.time()

    # Let's try using the minimum loss found during the
    # optimization
    losses = []

    # Train the model
    model.train()
    for epoch in range(epochs):
        # Zero out the gradients
        optimizer.zero_grad()

        # The loss is the sum of the compliance
        outputs = model(inputs=model_inputs)
        loss = objective(outputs)

        # Go through the backward pass and create the gradients
        loss.backward()
        losses.append(loss.item())

        # Step through the optimzer to update the data with the gradients
        optimizer.step()
        scheduler.step(loss)

        iterations += 1

    # Maximum iterations reached
    if iterations >= len(range(epochs)):
        termination_code = 4

    # Get the final results
    end = time.time()
    total_time = end - start  # noqa

    # Get final values
    outputs = model(inputs=model_inputs)
    f_final = objective(outputs)  # noqa
    f_final = f_final.detach().cpu().numpy()  # noqa

    # F-min
    f_min = np.min(np.array(losses))  # noqa

    results = {
        'f_init': f_init.detach().cpu().numpy(),
        'total_time': total_time,
        'f_final': f_final,
        'iterations': iterations,
        'fn_evals': None,  # Does not apply to this method
        'termination_code': termination_code,
        'objective_values': np.array(losses),
    }

    return results


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
