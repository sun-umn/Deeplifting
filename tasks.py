#!/usr/bin/python
# stdlib
import json
import os
import time
import warnings
from datetime import datetime
from functools import partial
from itertools import product
from multiprocessing import Pool, cpu_count

# third party
import click
import neptune
import numpy as np
import pandas as pd
import torch
import tqdm
import wandb

# first party
from config import (
    ackley_series,
    alpine_series,
    chung_reynolds_series,
    griewank_series,
    lennard_jones_series,
    levy_series,
    qing_series,
    rastrigin_series,
    schwefel_series,
)
from deeplifting.models import ReLUDeepliftingMLP
from deeplifting.optimization import run_pyomo  # noqa
from deeplifting.optimization import (
    run_adam_deeplifting,
    run_basinhopping,
    run_differential_evolution,
    run_dual_annealing,
    run_lbfgs_deeplifting,
    run_pygranso,
    run_pygranso_deeplifting,
    run_sgd_deeplifting,
)
from deeplifting.problems import HIGH_DIMENSIONAL_PROBLEMS_BY_NAME, PROBLEMS_BY_NAME
from deeplifting.utils import Results, get_devices, initialize_vector, set_seed

# Filter warnings
warnings.filterwarnings('ignore')


@click.group()
def cli():  # noqa
    pass


# Dual Annealing
@cli.command('run-dual-annealing-task')
@click.option('--problem_name', default='ackley')
@click.option('--dimensionality', default='low-dimensional')
@click.option('--experimentation', default=True)
def run_dual_annealing_task(
    problem_name: str, dimensionality: str, experimentation: bool
) -> None:
    """
    Function to run the dual annealing task for a single
    problem
    """
    # Setup the problem
    if dimensionality == 'low-dimensional':
        directory = 'low-dimension'
        PROBLEMS = PROBLEMS_BY_NAME
        API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
        PROJECT_NAME = 'Deeplifting-LD'

    elif dimensionality == 'high-dimensional':
        directory = 'high-dimension'
        PROBLEMS = HIGH_DIMENSIONAL_PROBLEMS_BY_NAME
        API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
        PROJECT_NAME = 'Deeplifting-HD'

    else:
        raise ValueError(f'{dimensionality} is not valid!')

    if experimentation:
        # Enable wandb
        wandb.login(key=API_KEY)

        wandb.init(
            # set the wandb project where this run will be logged
            project=PROJECT_NAME,
            tags=['dual-annealing', f'{problem_name}'],
        )

    print(f'Dual-Annealing for {problem_name}')

    # Create the save path for this task
    save_path = os.path.join(
        '/home/jusun/dever120/Deeplifting',
        'experiments/3b39b4fb-0520-4795-aaba-a8eab24ff8fd/',
        f'{directory}/dual-annealing',
    )

    # Setup the problem
    problem = PROBLEMS[problem_name]

    # Get the known minimum
    global_minimum = problem['global_minimum']

    # Get the number of trails
    trials = 50

    # Max iterations search space
    maxiters_space = [100, 500, 750, 1000, 5000, 10000]
    init_temp_space = [1, 500, 1000, 5000, 7500, 10000]

    # Next add dual annealing
    parameters = list(product(maxiters_space, init_temp_space))

    # Run dual annealing for different parameters
    dual_annealing_fn = partial(run_dual_annealing, problem=problem)
    dual_annleaing_results_list = []
    for maxiter, init_temp in tqdm.tqdm(parameters):
        print(f'Running dual-annealing parameters: maxiter={maxiter}; ')
        print(f'init_temp={init_temp}')
        dual_annealing_outputs = dual_annealing_fn(
            trials=trials, maxiter=maxiter, init_temp=init_temp
        )
        dual_annleaing_results_list.append(dual_annealing_outputs)

    # Concat all results
    dual_annleaing_results = pd.concat(dual_annleaing_results_list)
    dual_annleaing_results['global_minimum'] = global_minimum

    # Compute the success rate
    numerator = np.abs(
        dual_annleaing_results['f_final'] - dual_annleaing_results['global_minimum']
    )
    denominator = np.abs(
        dual_annleaing_results['f_init'] - dual_annleaing_results['global_minimum']
    )

    # Set up success
    dual_annleaing_results['success'] = ((numerator / denominator) <= 1e-4).astype(int)

    # Save the results
    save_file_name = os.path.join(save_path, f'{problem_name}-dual-annealing.parquet')
    dual_annleaing_results.to_parquet(save_file_name)

    print('Task completed! 🎉')


# Basinhopping
@cli.command('run-basinhopping-task')
@click.option('--problem_name', default='ackley')
@click.option('--dimensionality', default='low-dimensional')
@click.option('--experimentation', default=True)
def run_basinhopping_task(
    problem_name: str, dimensionality: str, experimentation: bool
) -> None:
    """
    Function to run the basinhopping task for a single
    problem
    """
    # Setup the problem
    if dimensionality == 'low-dimensional':
        directory = 'low-dimension'
        PROBLEMS = PROBLEMS_BY_NAME
        API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
        PROJECT_NAME = 'Deeplifting-LD'

    elif dimensionality == 'high-dimensional':
        directory = 'high-dimension'
        PROBLEMS = HIGH_DIMENSIONAL_PROBLEMS_BY_NAME
        API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
        PROJECT_NAME = 'Deeplifting-HD'

    else:
        raise ValueError(f'{dimensionality} is not valid!')

    if experimentation:
        # Enable wandb
        wandb.login(key=API_KEY)

        wandb.init(
            # set the wandb project where this run will be logged
            project=PROJECT_NAME,
            tags=['basinhopping', f'{problem_name}'],
        )

    print(f'Basinhopping for {problem_name}')

    # Create the save path for this task
    save_path = os.path.join(
        '/home/jusun/dever120/Deeplifting',
        'experiments/3b39b4fb-0520-4795-aaba-a8eab24ff8fd/',
        f'{directory}/basinhopping',
    )

    # Setup the problem
    problem = PROBLEMS[problem_name]

    # Get the known minimum
    global_minimum = problem['global_minimum']

    # Get the number of trails
    trials = 50

    # Max iterations search space
    maxiters_space = [100, 500, 750, 1000, 5000, 10000]
    T_space = [0.5, 1.0, 2.5, 5.0, 10.0]

    # Next add dual annealing
    parameters = list(product(maxiters_space, T_space))

    # Run basinhopping for different parameters
    basinhopping_fn = partial(run_basinhopping, problem=problem)
    basinhopping_results_list = []
    for niter, T in tqdm.tqdm(parameters):
        print(f'Running basinhopping parameters: niter={niter}; T={T}')
        basinhopping_outputs = basinhopping_fn(trials=trials, niter=niter, T=T)
        basinhopping_results_list.append(basinhopping_outputs)

    # Concat all results
    basinhopping_results = pd.concat(basinhopping_results_list)
    basinhopping_results['global_minimum'] = global_minimum

    # Compute the success rate
    numerator = np.abs(
        basinhopping_results['f_final'] - basinhopping_results['global_minimum']
    )
    denominator = np.abs(
        basinhopping_results['f_init'] - basinhopping_results['global_minimum']
    )

    # Set up success
    basinhopping_results['success'] = ((numerator / denominator) <= 1e-4).astype(int)

    # Save the results
    save_file_name = os.path.join(save_path, f'{problem_name}-basinhopping.parquet')
    basinhopping_results.to_parquet(save_file_name)

    print('Task completed! 🎉')


# Differential Evolution
@cli.command('run-differential-evolution-task')
@click.option('--problem_name', default='ackley')
@click.option('--dimensionality', default='low-dimensional')
@click.option('--experimentation', default=True)
def run_differential_evolution_task(
    problem_name: str, dimensionality: str, experimentation: bool
) -> None:
    """
    Function to run the differential evolution task for a single
    problem
    """
    # Setup the problem
    if dimensionality == 'low-dimensional':
        directory = 'low-dimension'
        PROBLEMS = PROBLEMS_BY_NAME
        API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
        PROJECT_NAME = 'Deeplifting-LD'

    elif dimensionality == 'high-dimensional':
        directory = 'high-dimension'
        PROBLEMS = HIGH_DIMENSIONAL_PROBLEMS_BY_NAME
        API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
        PROJECT_NAME = 'Deeplifting-HD'

    else:
        raise ValueError(f'{dimensionality} is not valid!')

    if experimentation:
        # Enable wandb
        wandb.login(key=API_KEY)

        wandb.init(
            # set the wandb project where this run will be logged
            project=PROJECT_NAME,
            tags=['differential-evolution', f'{problem_name}'],
        )

    print(f'Differential-Evolution for {problem_name}')

    # Create the save path for this task
    save_path = os.path.join(
        '/home/jusun/dever120/Deeplifting',
        'experiments/3b39b4fb-0520-4795-aaba-a8eab24ff8fd/',
        f'{directory}/differential-evolution',
    )

    # Setup the problem
    problem = PROBLEMS[problem_name]

    # Get the known minimum
    global_minimum = problem['global_minimum']

    # Get the number of trails
    trials = 50

    # Max iterations search space
    maxiters_space = [500, 750, 1000, 5000, 10000]
    popsize_space = [15, 20, 35, 50]
    mutation_space = [(0.5, 1.0), (0.5, 1.5), (0.5, 1.90)]
    recombination_space = [0.7, 0.5, 0.1]

    # Next add dual annealing
    parameters = list(
        product(maxiters_space, popsize_space, mutation_space, recombination_space)
    )

    # Run differential evolution for different parameters
    differential_evolution_fn = partial(run_differential_evolution, problem=problem)
    differential_evolution_results_list = []
    for maxiter, popsize, mutation, recombination in tqdm.tqdm(parameters):
        print('Running differential evoluation parameters: ')
        print(f'maxiter={maxiter}; popsize={popsize}; mutation={mutation}; ')
        print(f'recombination={recombination}')
        differential_evolution_outputs = differential_evolution_fn(
            trials=trials,
            maxiter=maxiter,
            popsize=popsize,
            mutation=mutation,
            recombination=recombination,
        )
        differential_evolution_results_list.append(differential_evolution_outputs)

    # Concat all results
    differential_evolution_results = pd.concat(differential_evolution_results_list)
    differential_evolution_results['global_minimum'] = global_minimum

    # Compute the success rate
    numerator = np.abs(
        differential_evolution_results['f_final']
        - differential_evolution_results['global_minimum']
    )
    denominator = np.abs(
        differential_evolution_results['f_init']
        - differential_evolution_results['global_minimum']
    )

    # Set up success
    differential_evolution_results['success'] = (
        (numerator / denominator) <= 1e-4
    ).astype(int)

    # Save the results
    save_file_name = os.path.join(
        save_path, f'{problem_name}-differential-evolution.parquet'
    )
    differential_evolution_results.to_parquet(save_file_name)

    print('Task completed! 🎉')


# # IPOPT
# @cli.command('run-ipopt-task')
# @click.option('--problem_name', default='ackley')
# @click.option('--dimensionality', default='low-dimensional')
# @click.option('--experimentation', default=True)
# def run_ipopt_task(
#     problem_name: str, dimensionality: str, experimentation: bool
# ) -> None:
#     """
#     Function to run the IPOPT task for a single
#     problem
#     """
#     # Setup the problem
#     if dimensionality == 'low-dimensional':
#         directory = 'low-dimension'
#         PROBLEMS = PROBLEMS_BY_NAME
#         API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
#         PROJECT_NAME = 'Deeplifting-LD'

#     elif dimensionality == 'high-dimensional':
#         directory = 'high-dimension'
#         PROBLEMS = HIGH_DIMENSIONAL_PROBLEMS_BY_NAME
#         API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
#         PROJECT_NAME = 'Deeplifting-HD'

#     else:
#         raise ValueError(f'{dimensionality} is not valid!')

#     if experimentation:
#         # Enable wandb
#         wandb.login(key=API_KEY)

#         wandb.init(
#             # set the wandb project where this run will be logged
#             project=PROJECT_NAME,
#             tags=['IPOPT', f'{problem_name}'],
#         )

#     print(f'IPOPT for {problem_name}')

#     # Create the save path for this task
#     save_path = os.path.join(
#         '/home/jusun/dever120/Deeplifting',
#         'experiments/3b39b4fb-0520-4795-aaba-a8eab24ff8fd/',
#         f'{directory}/ipopt',
#     )

#     # Setup the problem
#     problem = PROBLEMS[problem_name]

#     # Get the known minimum
#     global_minimum = problem['global_minimum']

#     # Get the number of trails
#     trials = 50

#     # Run ipopt
#     ipopt_results = run_ipopt(
#         problem=problem,
#         trials=trials,
#     )

#     ipopt_results['global_minimum'] = global_minimum

#     # Compute the success rate
#     numerator = np.abs(ipopt_results['f_final'] - ipopt_results['global_minimum'])
#     denominator = np.abs(ipopt_results['f_init'] - ipopt_results['global_minimum'])

#     # Set up success
#     ipopt_results['success'] = ((numerator / denominator) <= 1e-4).astype(int)

#     # Save the results
#     save_file_name = os.path.join(save_path, f'{problem_name}-ipopt.parquet')
#     ipopt_results.to_parquet(save_file_name)

#     print('Task completed! 🎉')


@cli.command('run-pygranso-task')
@click.option('--problem_name', default='ackley')
@click.option('--dimensionality', default='low-dimensional')
@click.option('--experimentation', default=True)
def run_pygranso_task(
    problem_name: str, dimensionality: str, experimentation: bool
) -> None:
    """
    Function that will run the competing algorithms to Deeplifting
    which is our NCVX method PyGranso
    """
    # Setup the problem
    if dimensionality == 'low-dimensional':
        directory = 'low-dimension'
        PROBLEMS = PROBLEMS_BY_NAME
        API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
        PROJECT_NAME = 'Deeplifting-LD'

    elif dimensionality == 'high-dimensional':
        directory = 'high-dimension'
        PROBLEMS = HIGH_DIMENSIONAL_PROBLEMS_BY_NAME
        API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
        PROJECT_NAME = 'Deeplifting-HD'

    else:
        raise ValueError(f'{dimensionality} is not valid!')

    if experimentation:
        # Enable wandb
        wandb.login(key=API_KEY)

        wandb.init(
            # set the wandb project where this run will be logged
            project=PROJECT_NAME,
            tags=['PyGranso', f'{problem_name}'],
        )

    print(f'PyGranso for {problem_name}')

    # Create the save path for this task
    save_path = os.path.join(
        '/home/jusun/dever120/Deeplifting',
        'experiments/3b39b4fb-0520-4795-aaba-a8eab24ff8fd/',
        f'{directory}/pygranso',
    )

    # Setup the problem
    problem = PROBLEMS[problem_name]

    # Get the known minimum
    global_minimum = problem['global_minimum']

    # Get the number of trails
    trials = 50

    # Run ipopt
    pygranso_results = run_pygranso(
        problem=problem,
        trials=trials,
    )

    pygranso_results['global_minimum'] = global_minimum

    # Compute the success rate
    numerator = np.abs(pygranso_results['f_final'] - pygranso_results['global_minimum'])
    denominator = np.abs(
        pygranso_results['f_init'] - pygranso_results['global_minimum']
    )

    # Set up success
    pygranso_results['success'] = ((numerator / denominator) <= 1e-4).astype(int)

    # Save the results
    save_file_name = os.path.join(save_path, f'{problem_name}-pygranso.parquet')
    pygranso_results.to_parquet(save_file_name)

    print('Task completed! 🎉')


@cli.command('run-algorithm-comparisons-scip')
@click.option('--problem_series', default='ackley')
@click.option('--dimensionality', default='low-dimensional')
@click.option('--trials', default=10)
def run_scip_task(problem_series, dimensionality, trials):
    """
    Function that will run the competing algorithms to Deeplifting.
    The current competitor models are:
    1. SCIP
    SCIP in general is fairly quick but it started to run slowly for some
    problems so I am making it its own command
    """
    print('Run Algorithms!')
    run = neptune.init_run(  # noqa
        project="dever120/Deeplifting",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzYmIwMTUyNC05YmZmLTQ1NzctOTEyNS1kZTIxYjU5NjY5YjAifQ==",  # noqa
    )  # your credentials
    run['sys/tags'].add(['scip', dimensionality])

    print('Run Algorithms!')
    # Get the problem list
    if dimensionality == 'high-dimensional':
        PROBLEMS = HIGH_DIMENSIONAL_PROBLEMS_BY_NAME
        if problem_series == 'ackley':
            problem_names = ackley_series
        elif problem_series == 'alpine1':
            problem_names = alpine_series
        elif problem_series == 'chung_reynolds':
            problem_names = chung_reynolds_series
        elif problem_series == 'griewank':
            problem_names = griewank_series
        elif problem_series == 'lennard_jones':
            problem_names = lennard_jones_series
        elif problem_series == 'levy':
            problem_names = levy_series
        elif problem_series == 'qing':
            problem_names = qing_series
        elif problem_series == 'rastrigin':
            problem_names = rastrigin_series
        elif problem_series == 'schwefel':
            problem_names = schwefel_series
    elif dimensionality == 'low-dimensional':
        if problem_series != 'all':
            raise ValueError('Can only run full list for this option!')
        PROBLEMS = PROBLEMS_BY_NAME

    # One experiment date
    experiment_date = datetime.today().strftime('%Y-%m-%d-%H')

    for problem_name in problem_names:
        print(problem_name)
        problem_performance_list = []

        # Setup the problem
        problem = PROBLEMS[problem_name]

        # Get the known minimum
        minimum_value = problem['global_minimum']

        # Get the dimensions
        dimensions = problem['dimensions']

        # Create column names
        x_columns = [f'x{i + 1}' for i in range(dimensions)]
        columns = x_columns + ['f', 'algorithm', 'time']

        # Next we need to implement the SCIP algorithm
        print('Running SCIP!')
        outputs_scip = run_pyomo(problem, trials=trials, method='scip')

        # Get the final results for all differential evolution runs
        scip_results = pd.DataFrame(outputs_scip['final_results'], columns=columns)
        scip_results['problem_name'] = problem_name
        scip_results['hits'] = np.where(
            np.abs(scip_results['f'] - minimum_value) <= 1e-4, 1, 0
        )
        scip_results['dimensions'] = dimensions

        # Print the results
        hits = scip_results['hits'].mean()
        average_time = scip_results['time'].mean()
        print(f'Success Rate = {hits}')
        print(f'Average time = {average_time}')

        # Add differential evolution to the problem_performance_list
        problem_performance_list.append(scip_results)

        # Concatenate all of the data at the end of each problem because
        # we can save intermediate results
        problem_performance_df = pd.concat(problem_performance_list, ignore_index=True)
        path = f'./algorithm_compare_results/{dimensionality}/{experiment_date}-{problem_name}-scip'  # noqa
        if not os.path.exists(path):
            os.makedirs(path)

        problem_performance_df.to_parquet(f'{path}/{dimensionality}.parquet')


@cli.command('find-best-deeplifting-architecture-v2')
@click.option('--problem_name', default='ackley')
@click.option('--method', default='deeplifting-pygranso')
@click.option('--dimensionality', default='low-dimensional')
@click.option('--experimentation', default=True)
@click.option('--include_weight_initialization', default=True)
def find_best_architecture_task(
    problem_name, method, dimensionality, experimentation, include_weight_initialization
):
    """
    Function that we will use to find the best architecture over multiple
    "hard" high-dimensional problems. We will aim to tackle a large dimensional
    space with this function, 500+
    """
    # Set the number of threads to 1
    os.environ['OMP_NUM_THREADS'] = '1'

    # Get the available device
    device = get_devices()

    # Weight initialization rounds
    # This will create 25 network initializations
    # for each point and we can study the variance
    max_weight_trials = {
        False: range(10, 20, 10),
        True: range(10, 110, 10),
    }

    # Setup the problem
    if dimensionality == 'low-dimensional':
        directory = 'low-dimension'
        PROBLEMS = PROBLEMS_BY_NAME
        API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
        PROJECT_NAME = 'Deeplifting-LD'

    elif dimensionality == 'high-dimensional':
        directory = 'high-dimension'
        PROBLEMS = HIGH_DIMENSIONAL_PROBLEMS_BY_NAME
        API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
        PROJECT_NAME = 'Deeplifting-HD'

    else:
        raise ValueError(f'{dimensionality} is not valid!')

    if experimentation:
        # Enable wandb
        wandb.login(key=API_KEY)

        wandb.init(
            # set the wandb project where this run will be logged
            project=PROJECT_NAME,
            tags=[f'{method}', f'{problem_name}'],
        )

    # Main directory to save data
    # This will be different for each user
    save_path = os.path.join(
        '/home/jusun/dever120/Deeplifting',
        'experiments/3b39b4fb-0520-4795-aaba-a8eab24ff8fd/',
        f'{directory}/deeplifting-pygranso',
    )

    # Get the problem information
    problem = PROBLEMS[problem_name]

    # Objective function
    objective = problem['objective']

    # Set up the function with pytorch option
    fn = lambda x: objective(x, version='pytorch')  # noqa

    # Bounds
    bounds = problem['bounds']

    # Get the device (CPU for now)
    output_size = problem['dimensions']

    # Maximum iterations for a problem
    # Most problems converge quickly but some
    # take a little longer
    max_iterations = problem['max_iterations']

    # Get the maximum number of trials
    # for the problem
    trials = problem['trials']

    # Get the global minimum
    global_minimum = problem['global_minimum']

    # Layer search
    layers = [2, 3, 4, 5]

    # Number of neurons
    units_search = [192, 128, 64, 32]

    # NOTE: Breakthrough that the size of the input dimension
    # has a direct impact on the models ability to find a global
    # solution so we will investigate this as well
    input_dimensions = [1, 2, 16, 32]

    # Initial layer type
    initial_layer_type = 'linear'
    include_bn = True
    learning_rates = [1.0]  # Define defualt Pygranso learning rate to be 1.0

    # Configs
    configuration = product(learning_rates, layers, units_search, input_dimensions)

    # Start the optimization process
    for lr, num_layers, units, input_dimension in configuration:
        # n-layer m neuron network
        hidden_sizes = (units,) * num_layers

        # Problem configuration in a dict
        problem_config = {
            'num_layers': num_layers,
            'num_neurons': units,
            'lr': lr,
            'input_dimension': input_dimension,
        }

        # Setup list to store information
        results = Results(method=method)

        # We have an observation that we can start at the same point
        # but it may or may not converge so we can try different
        # weights
        for index, trial in enumerate(range(trials)):
            # Set the seed
            set_seed(trial)

            # Fix the inputs for deeplifting
            if initial_layer_type == 'embedding':
                inputs = torch.randint(
                    low=0, high=(units - 1), size=(input_dimension, units)
                )
                inputs = inputs.to(device=device, dtype=torch.long)

            elif initial_layer_type == 'linear':
                inputs = torch.randn(size=(input_dimension, 5 * output_size))
                inputs = inputs.to(device=device, dtype=torch.double)

            else:
                raise ValueError(f'{initial_layer_type} is not an accepted type!')

            # Initialization for other models
            x_start = initialize_vector(size=output_size, bounds=bounds)

            # Let's also put the starting distance
            distance = np.mean((x_start - problem['global_x']) ** 2)

            x_start = torch.tensor(x_start)
            x_start = x_start.to(device=device, dtype=torch.double)

            # Build the xs for the outputs
            columns = [f'x{i + 1}' for i in range(output_size)]
            xs = json.dumps(dict(zip(columns, x_start.detach().cpu().numpy())))

            # Creates different weight intializations for the same starting point
            # x0
            for i in max_weight_trials[include_weight_initialization]:
                seed = (i + index) * i
                print(f'Fitting point {x_start} - with weights {i}')
                print(f' - layers - {num_layers} - units - {units} - trial - {trial}')
                print(f' - input dimension {input_dimension} - seed = {seed}')

                # Deeplifting model with skip connections
                model = ReLUDeepliftingMLP(
                    initial_hidden_size=(5 * output_size),
                    hidden_sizes=hidden_sizes,
                    output_size=output_size,
                    bounds=bounds,
                    initial_layer_type=initial_layer_type,
                    include_weight_initialization=include_weight_initialization,
                    include_bn=include_bn,
                    seed=seed,
                )

                model = model.to(device=device, dtype=torch.double)

                if method == 'deeplifting-pygranso':
                    # Run PyGranso Based Deeplifting
                    deeplifting_outputs = run_pygranso_deeplifting(
                        model=model,
                        model_inputs=inputs,
                        start_position=x_start,
                        objective=fn,
                        device=device,
                        max_iterations=max_iterations,
                    )

                elif method == 'deeplifting-lbfgs':
                    # Run LBFGS Based Deeplifting
                    deeplifting_outputs = run_lbfgs_deeplifting(
                        model=model,
                        model_inputs=inputs,
                        start_position=x_start,
                        objective=fn,
                        device=device,
                        max_iterations=max_iterations,
                    )

                elif method == 'deeplifting-adam':
                    # Run LBFGS Based Deeplifting
                    deeplifting_outputs = run_adam_deeplifting(
                        model=model,
                        model_inputs=inputs,
                        start_position=x_start,
                        objective=fn,
                        device=device,
                        max_iterations=max_iterations,
                    )

                else:
                    raise ValueError(f'Method {method} is not a valid option!')

                # Unpack results
                f_init = deeplifting_outputs.get('f_init')
                f_final = deeplifting_outputs.get('f_final')
                total_time = deeplifting_outputs.get('total_time')
                iterations = deeplifting_outputs.get('iterations')
                fn_evals = deeplifting_outputs.get('fn_evals')
                termination_code = deeplifting_outputs.get('termination_code')
                objective_values = deeplifting_outputs.get('objective_values')

                # Append results
                results.append_record(
                    global_minimum=global_minimum,
                    f_init=f_init,
                    f_final=f_final,
                    total_time=total_time,
                    iterations=iterations,
                    fn_evals=fn_evals,
                    termination_code=termination_code,
                    problem_config=problem_config,
                    xs=xs,
                    method=method,
                    lr=lr,
                    objective_values=objective_values,
                    distance=distance,
                )

        # Create the data from this run and save sequentially
        results.build_and_save_dataframe(save_path=save_path, problem_name=problem_name)


@cli.command('find-best-deeplifting-architecture-sgd')
@click.option('--problem_name', default='ackley')
@click.option('--dimensionality', default='low-dimensional')
@click.option('--experimentation', default=True)
def find_best_architecture_sgd_task(
    problem_name,
    dimensionality,
    experimentation,
):
    """
    Function that we will use to find the best architecture over multiple
    "hard" high-dimensional problems. We will aim to tackle a large dimensional
    space with this function, 500+
    """
    # Set the number of threads to 1
    os.environ['OMP_NUM_THREADS'] = '1'

    # Method
    method = 'deeplifting-sgd'

    # Get the available device
    device = get_devices()

    # Include weight initialization
    include_weight_initialization = True

    # Weight initialization rounds
    # This will create 25 network initializations
    # for each point and we can study the variance
    max_weight_trials = {
        False: range(10, 20, 10),
        True: range(10, 110, 10),
    }

    # Setup the problem
    if dimensionality == 'low-dimensional':
        directory = 'low-dimension'
        PROBLEMS = PROBLEMS_BY_NAME
        API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
        PROJECT_NAME = 'Deeplifting-LD'

    elif dimensionality == 'high-dimensional':
        directory = 'high-dimension'
        PROBLEMS = HIGH_DIMENSIONAL_PROBLEMS_BY_NAME
        API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
        PROJECT_NAME = 'Deeplifting-HD'

    else:
        raise ValueError(f'{dimensionality} is not valid!')

    if experimentation:
        # Enable wandb
        wandb.login(key=API_KEY)

        wandb.init(
            # set the wandb project where this run will be logged
            project=PROJECT_NAME,
            tags=[f'{method}', f'{problem_name}'],
        )

    # Main directory to save data
    # This will be different for each user
    save_path = os.path.join(
        '/home/jusun/dever120/Deeplifting',
        'experiments/3b39b4fb-0520-4795-aaba-a8eab24ff8fd/',
        f'{directory}/deeplifting-sgd',
    )

    # Get the problem information
    problem = PROBLEMS[problem_name]

    # Objective function
    objective = problem['objective']

    # Set up the function with pytorch option
    fn = lambda x: objective(x, version='pytorch')  # noqa

    # Bounds
    bounds = problem['bounds']

    # Get the device (CPU for now)
    output_size = problem['dimensions']

    # Get the maximum number of trials
    # for the problem
    trials = problem['trials']

    # Get the global minimum
    global_minimum = problem['global_minimum']

    # Layer search
    layers = [2, 3, 4, 5]

    # Number of neurons
    units_search = [192, 128, 64, 32]

    # NOTE: Breakthrough that the size of the input dimension
    # has a direct impact on the models ability to find a global
    # solution so we will investigate this as well
    input_dimensions = [1, 2, 16, 32]

    # Initial layer type
    initial_layer_type = 'linear'
    include_bn = True
    learning_rates = [1e-2]

    # Configs
    configuration = product(learning_rates, layers, units_search, input_dimensions)

    # Start the optimization process
    for lr, num_layers, units, input_dimension in configuration:
        # n-layer m neuron network
        hidden_sizes = (units,) * num_layers

        # Problem configuration in a dict
        problem_config = {
            'num_layers': num_layers,
            'num_neurons': units,
            'lr': lr,
            'input_dimension': input_dimension,
        }

        # Setup list to store information
        results = Results(method=method)

        # We have an observation that we can start at the same point
        # but it may or may not converge so we can try different
        # weights
        for index, trial in enumerate(range(trials)):
            # Set the seed
            set_seed(trial)

            # Fix the inputs for deeplifting
            if initial_layer_type == 'embedding':
                inputs = torch.randint(
                    low=0, high=(units - 1), size=(input_dimension, units)
                )
                inputs = inputs.to(device=device, dtype=torch.long)

            elif initial_layer_type == 'linear':
                inputs = torch.rand(size=(input_dimension, 5 * output_size))
                inputs = inputs.to(device=device, dtype=torch.double)

            else:
                raise ValueError(f'{initial_layer_type} is not an accepted type!')

            # Initialization for other models
            x_start = initialize_vector(size=output_size, bounds=bounds)

            # Let's also put the starting distance
            distance = np.mean((x_start - problem['global_x']) ** 2)

            x_start = torch.tensor(x_start)
            x_start = x_start.to(device=device, dtype=torch.double)

            # Build the xs for the outputs
            columns = [f'x{i + 1}' for i in range(output_size)]
            xs = json.dumps(dict(zip(columns, x_start.detach().cpu().numpy())))

            # Creates different weight intializations
            # for the same starting point x0
            for i in max_weight_trials[include_weight_initialization]:
                seed = (i + index) * i
                print(f'Fitting point {x_start} - with weights {i}')
                print(
                    f' - layers - {num_layers} - units'
                    f' - {units} - trial - {trial}'
                    f' - lr {lr}'
                )
                print(f'seed = {seed}')

                # Deeplifting model with skip connections
                model = ReLUDeepliftingMLP(
                    initial_hidden_size=(5 * output_size),
                    hidden_sizes=hidden_sizes,
                    output_size=output_size,
                    bounds=bounds,
                    initial_layer_type=initial_layer_type,
                    include_weight_initialization=include_weight_initialization,
                    include_bn=include_bn,
                    seed=seed,
                )

                model = model.to(device=device, dtype=torch.double)

                # Run PyGranso Based Deeplifting
                deeplifting_outputs = run_sgd_deeplifting(
                    model=model,
                    model_inputs=inputs,
                    start_position=x_start,
                    objective=fn,
                    device=device,
                    lr=lr,
                )

                # Unpack results
                f_init = deeplifting_outputs.get('f_init')
                f_final = deeplifting_outputs.get('f_final')
                total_time = deeplifting_outputs.get('total_time')
                iterations = deeplifting_outputs.get('iterations')
                fn_evals = deeplifting_outputs.get('fn_evals')
                termination_code = deeplifting_outputs.get('termination_code')
                objective_values = deeplifting_outputs.get('objective_values')

                # Append results
                results.append_record(
                    global_minimum=global_minimum,
                    f_init=f_init,
                    f_final=f_final,
                    total_time=total_time,
                    iterations=iterations,
                    fn_evals=fn_evals,
                    termination_code=termination_code,
                    problem_config=problem_config,
                    xs=xs,
                    method=method,
                    lr=lr,
                    objective_values=objective_values,
                    distance=distance,
                )

        # Create the data from this run and save sequentially
        results.build_and_save_dataframe(save_path=save_path, problem_name=problem_name)


@cli.command('find-best-deeplifting-architecture-adam')
@click.option('--problem_name', default='ackley')
@click.option('--dimensionality', default='low-dimensional')
@click.option('--experimentation', default=True)
def find_best_architecture_adam_task(
    problem_name,
    dimensionality,
    experimentation,
):
    """
    Function that we will use to find the best architecture over multiple
    "hard" high-dimensional problems. We will aim to tackle a large dimensional
    space with this function, 500+
    """
    # Set the number of threads to 1
    os.environ['OMP_NUM_THREADS'] = '1'

    # Method
    method = 'deeplifting-adam'

    # Get the available device
    device = get_devices()

    # Include weight initialization
    include_weight_initialization = True

    # Weight initialization rounds
    # This will create 25 network initializations
    # for each point and we can study the variance
    max_weight_trials = {
        False: range(10, 20, 10),
        True: range(10, 110, 10),
    }

    # Setup the problem
    if dimensionality == 'low-dimensional':
        directory = 'low-dimension'
        PROBLEMS = PROBLEMS_BY_NAME
        API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
        PROJECT_NAME = 'Deeplifting-LD'

    elif dimensionality == 'high-dimensional':
        directory = 'high-dimension'
        PROBLEMS = HIGH_DIMENSIONAL_PROBLEMS_BY_NAME
        API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
        PROJECT_NAME = 'Deeplifting-HD'

    else:
        raise ValueError(f'{dimensionality} is not valid!')

    if experimentation:
        # Enable wandb
        wandb.login(key=API_KEY)

        wandb.init(
            # set the wandb project where this run will be logged
            project=PROJECT_NAME,
            tags=[f'{method}', f'{problem_name}'],
        )

    # Main directory to save data
    # This will be different for each user
    save_path = os.path.join(
        '/home/jusun/dever120/Deeplifting',
        'experiments/3b39b4fb-0520-4795-aaba-a8eab24ff8fd/',
        f'{directory}/deeplifting-adam',
    )

    # Get the problem information
    problem = PROBLEMS[problem_name]

    # Objective function
    objective = problem['objective']

    # Set up the function with pytorch option
    fn = lambda x: objective(x, version='pytorch')  # noqa

    # Bounds
    bounds = problem['bounds']

    # Get the device (CPU for now)
    output_size = problem['dimensions']

    # Get the maximum number of trials
    # for the problem
    trials = problem['trials']

    # Get the global minimum
    global_minimum = problem['global_minimum']

    # Layer search
    layers = [2, 3, 4, 5]

    # Number of neurons
    units_search = [192, 128, 64, 32]

    # NOTE: Breakthrough that the size of the input dimension
    # has a direct impact on the models ability to find a global
    # solution so we will investigate this as well
    input_dimensions = [1, 2, 16, 32]

    # Initial layer type
    initial_layer_type = 'linear'
    include_bn = True
    learning_rates = [1.0, 1e-1, 1e-2]

    # Configs
    configuration = product(learning_rates, layers, units_search, input_dimensions)

    # Start the optimization process
    for lr, num_layers, units, input_dimension in configuration:
        # n-layer m neuron network
        hidden_sizes = (units,) * num_layers

        # Problem configuration in a dict
        problem_config = {
            'num_layers': num_layers,
            'num_neurons': units,
            'lr': lr,
            'input_dimension': input_dimension,
        }

        # Setup list to store information
        results = Results(method=method)

        # We have an observation that we can start at the same point
        # but it may or may not converge so we can try different
        # weights
        for index, trial in enumerate(range(trials)):
            # Set the seed
            set_seed(trial)

            # Fix the inputs for deeplifting
            if initial_layer_type == 'embedding':
                inputs = torch.randint(
                    low=0, high=(units - 1), size=(input_dimension, units)
                )
                inputs = inputs.to(device=device, dtype=torch.long)

            elif initial_layer_type == 'linear':
                inputs = torch.rand(size=(input_dimension, 5 * output_size))
                inputs = inputs.to(device=device, dtype=torch.double)

            else:
                raise ValueError(f'{initial_layer_type} is not an accepted type!')

            # Initialization for other models
            x_start = initialize_vector(size=output_size, bounds=bounds)

            # Let's also put the starting distance
            distance = np.mean((x_start - problem['global_x']) ** 2)

            x_start = torch.tensor(x_start)
            x_start = x_start.to(device=device, dtype=torch.double)

            # Build the xs for the outputs
            columns = [f'x{i + 1}' for i in range(output_size)]
            xs = json.dumps(dict(zip(columns, x_start.detach().cpu().numpy())))

            # Creates different weight intializations
            # for the same starting point x0
            for i in max_weight_trials[include_weight_initialization]:
                seed = (i + index) * i
                print(f'Fitting point {x_start} - with weights {i}')
                print(
                    f' - layers - {num_layers} - units'
                    f' - {units} - trial - {trial}'
                    f' - lr {lr}'
                )
                print(f'seed = {seed}')

                # Deeplifting model with skip connections
                model = ReLUDeepliftingMLP(
                    initial_hidden_size=(5 * output_size),
                    hidden_sizes=hidden_sizes,
                    output_size=output_size,
                    bounds=bounds,
                    initial_layer_type=initial_layer_type,
                    include_weight_initialization=include_weight_initialization,
                    include_bn=include_bn,
                    seed=seed,
                )

                model = model.to(device=device, dtype=torch.double)

                # Run PyGranso Based Deeplifting
                deeplifting_outputs = run_adam_deeplifting(
                    model=model,
                    model_inputs=inputs,
                    start_position=x_start,
                    objective=fn,
                    device=device,
                    lr=lr,
                )

                # Unpack results
                f_init = deeplifting_outputs.get('f_init')
                f_final = deeplifting_outputs.get('f_final')
                total_time = deeplifting_outputs.get('total_time')
                iterations = deeplifting_outputs.get('iterations')
                fn_evals = deeplifting_outputs.get('fn_evals')
                termination_code = deeplifting_outputs.get('termination_code')
                objective_values = deeplifting_outputs.get('objective_values')

                # Append results
                results.append_record(
                    global_minimum=global_minimum,
                    f_init=f_init,
                    f_final=f_final,
                    total_time=total_time,
                    iterations=iterations,
                    fn_evals=fn_evals,
                    termination_code=termination_code,
                    problem_config=problem_config,
                    xs=xs,
                    method=method,
                    lr=lr,
                    objective_values=objective_values,
                    distance=distance,
                )

        # Create the data from this run and save sequentially
        results.build_and_save_dataframe(save_path=save_path, problem_name=problem_name)


def run_deeplifting_pygranso_parallel(inputs, debug=False):
    """
    Run the deeplifting-pygranso function in parallel
    """

    if debug:
        print(inputs)
        print('\n')
        print('Debugging 🐞')
        print('\n')

    else:
        # Inputs
        lr = inputs[0]
        num_layers = inputs[1]
        units = inputs[2]
        input_dimension = inputs[3]
        problem = inputs[4]
        save_path = inputs[5]

        # Get the available device
        device = get_devices()  # noqa

        # Include weight initialization
        include_weight_initialization = True  # noqa

        # Weight initialization rounds
        # This will create 25 network initializations
        # for each point and we can study the variance
        max_weight_trials = {  # noqa
            False: range(10, 20, 10),
            True: range(10, 110, 10),
        }

        # method
        method = 'deeplifting-pygranso'

        # Initial layer type
        initial_layer_type = 'linear'  # noqa
        include_bn = True  # noqa

        # Problem name
        problem_name = problem['name']

        # Objective function
        objective = problem['objective']

        # Set up the function with pytorch option
        fn = lambda x: objective(x, version='pytorch')  # noqa

        # Bounds
        bounds = problem['bounds']  # noqa

        # Get the device (CPU for now)
        output_size = problem['dimensions']  # noqa

        # Get the maximum number of trials
        # for the problem
        trials = problem['trials']  # noqa

        # Maximum iterations for a problem
        # Most problems converge quickly but some
        # take a little longer
        max_iterations = problem['max_iterations']

        # Get the global minimum
        global_minimum = problem['global_minimum']  # noqa

        # n-layer m neuron network
        hidden_sizes = (units,) * num_layers

        # Problem configuration in a dict
        problem_config = {
            'num_layers': num_layers,
            'num_neurons': units,
            'lr': lr,
            'input_dimension': input_dimension,
        }

        # Setup list to store information
        results = Results(method=method)

        # We have an observation that we can start at the same point
        # but it may or may not converge so we can try different
        # weights
        for index, trial in enumerate(range(trials)):
            # Set the seed
            set_seed(trial)

            # Fix the inputs for deeplifting
            if initial_layer_type == 'embedding':
                inputs = torch.randint(
                    low=0, high=(units - 1), size=(input_dimension, units)
                )
                inputs = inputs.to(device=device, dtype=torch.long)

            elif initial_layer_type == 'linear':
                inputs = torch.randn(size=(input_dimension, 5 * output_size))
                inputs = inputs.to(device=device, dtype=torch.double)

            else:
                raise ValueError(f'{initial_layer_type} is not an accepted type!')

            # Initialization for other models
            x_start = initialize_vector(size=output_size, bounds=bounds)

            # Let's also put the starting distance
            distance = np.mean((x_start - problem['global_x']) ** 2)

            x_start = torch.tensor(x_start)
            x_start = x_start.to(device=device, dtype=torch.double)

            # Build the xs for the outputs
            columns = [f'x{i + 1}' for i in range(output_size)]
            xs = json.dumps(dict(zip(columns, x_start.detach().cpu().numpy())))

            # Creates different weight intializations for the same starting point
            # x0
            for i in max_weight_trials[include_weight_initialization]:
                seed = (i + index) * i

                # Deeplifting model with skip connections
                model = ReLUDeepliftingMLP(
                    initial_hidden_size=(5 * output_size),
                    hidden_sizes=hidden_sizes,
                    output_size=output_size,
                    bounds=bounds,
                    initial_layer_type=initial_layer_type,
                    include_weight_initialization=include_weight_initialization,
                    include_bn=include_bn,
                    seed=seed,
                )

                model = model.to(device=device, dtype=torch.double)

                if method == 'deeplifting-pygranso':
                    # Run PyGranso Based Deeplifting
                    deeplifting_outputs = run_pygranso_deeplifting(
                        model=model,
                        model_inputs=inputs,
                        start_position=x_start,
                        objective=fn,
                        device=device,
                        max_iterations=max_iterations,
                    )

                elif method == 'deeplifting-lbfgs':
                    # Run LBFGS Based Deeplifting
                    deeplifting_outputs = run_lbfgs_deeplifting(
                        model=model,
                        model_inputs=inputs,
                        start_position=x_start,
                        objective=fn,
                        device=device,
                        max_iterations=max_iterations,
                    )

                elif method == 'deeplifting-adam':
                    # Run LBFGS Based Deeplifting
                    deeplifting_outputs = run_adam_deeplifting(
                        model=model,
                        model_inputs=inputs,
                        start_position=x_start,
                        objective=fn,
                        device=device,
                        max_iterations=max_iterations,
                    )

                else:
                    raise ValueError(f'Method {method} is not a valid option!')

                # Unpack results
                f_init = deeplifting_outputs.get('f_init')
                f_final = deeplifting_outputs.get('f_final')
                total_time = deeplifting_outputs.get('total_time')
                iterations = deeplifting_outputs.get('iterations')
                fn_evals = deeplifting_outputs.get('fn_evals')
                termination_code = deeplifting_outputs.get('termination_code')
                objective_values = deeplifting_outputs.get('objective_values')

                # Append results
                results.append_record(
                    global_minimum=global_minimum,
                    f_init=f_init,
                    f_final=f_final,
                    total_time=total_time,
                    iterations=iterations,
                    fn_evals=fn_evals,
                    termination_code=termination_code,
                    problem_config=problem_config,
                    xs=xs,
                    method=method,
                    lr=lr,
                    objective_values=objective_values,
                    distance=distance,
                )

        # Create the data from this run and save sequentially
        results.build_and_save_dataframe(save_path=save_path, problem_name=problem_name)
        print('Deeplifting completed! 🎉')


@cli.command('test-parallel')
@click.option('--problem_name', default='schwefel')
@click.option('--dimensionality', default='low-dimensional')
@click.option('--experimentation', default=True)
def test_parallel(
    problem_name,
    dimensionality,
    experimentation,
):
    """
    Function that we will use to find the best architecture over multiple
    "hard" high-dimensional problems. We will aim to tackle a large dimensional
    space with this function, 500+
    """
    print('Starting slurm output ⏳')
    os.environ['OMP_NUM_THREADS'] = '1'
    method = 'deeplifting-pygranso'

    # Setup the problem
    if dimensionality == 'low-dimensional':
        directory = 'low-dimension'  # noqa
        PROBLEMS = PROBLEMS_BY_NAME
        API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
        PROJECT_NAME = 'Deeplifting-LD'

    elif dimensionality == 'high-dimensional':
        directory = 'high-dimension'  # noqa
        PROBLEMS = HIGH_DIMENSIONAL_PROBLEMS_BY_NAME
        API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
        PROJECT_NAME = 'Deeplifting-HD'

    else:
        raise ValueError(f'{dimensionality} is not valid!')

    if experimentation:
        # Enable wandb
        wandb.login(key=API_KEY)

        wandb.init(
            # set the wandb project where this run will be logged
            project=PROJECT_NAME,
            tags=[f'{method}', f'{problem_name}'],
        )

    print('Starting job 🏁')
    print(f'Number of cpus {cpu_count()}')

    # Main directory to save data
    # This will be different for each user
    save_path = os.path.join(  # noqa
        '/home/jusun/dever120/Deeplifting',
        'experiments/3b39b4fb-0520-4795-aaba-a8eab24ff8fd/',
        f'{directory}/{method}/group',
    )
    save_path = [save_path]

    # Get the problem information
    problem = PROBLEMS[problem_name]
    problem = [problem]

    # Layer search
    layers = [2, 3, 4, 5, 7, 10]

    # Number of neurons
    units_search = [128, 64, 32, 16]

    # NOTE: Breakthrough that the size of the input dimension
    # has a direct impact on the models ability to find a global
    # solution so we will investigate this as well
    input_dimensions = [1, 2, 4, 8]

    # Learning rates
    learning_rates = [1.0]

    # Configs
    configuration = product(
        learning_rates, layers, units_search, input_dimensions, problem, save_path
    )
    config = list(configuration)
    print(f'There are {len(config)} configurations ✅')

    # Start ray process
    start = time.time()
    with Pool(8) as pool:
        for _ in tqdm.tqdm(pool.imap(run_deeplifting_pygranso_parallel, config)):
            pass
    end = time.time()
    print(f'Total time {end - start} ⏰')


if __name__ == "__main__":
    # Be able to run different commands
    cli()
