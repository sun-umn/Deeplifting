#!/usr/bin/python
# stdlib
import json
import os
import time
import warnings
from datetime import datetime

# third party
import click
import neptune
import numpy as np
import pandas as pd
import torch
import wandb
from pygranso.private.getNvar import getNvarTorch
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct

# first party
from config import (
    ackley_series,
    alpine_series,
    chung_reynolds_series,
    griewank_series,
    high_dimensional_problem_names,
    lennard_jones_series,
    levy_series,
    low_dimensional_problem_names,
    qing_series,
    rastrigin_series,
    schwefel_series,
)
from deeplifting.models import ReLUDeepliftingMLP
from deeplifting.optimization import run_pyomo  # noqa
from deeplifting.optimization import (
    deeplifting_nd_fn,
    deeplifting_predictions,
    run_basinhopping,
    run_differential_evolution,
    run_dual_annealing,
    run_ipopt,
    run_pygranso,
)
from deeplifting.problems import HIGH_DIMENSIONAL_PROBLEMS_BY_NAME, PROBLEMS_BY_NAME
from deeplifting.utils import get_devices, initialize_vector, train_model_to_output

# Filter warnings
warnings.filterwarnings('ignore')


@click.group()
def cli():
    pass


@cli.command('run-algorithm-comparisons')
@click.option('--dimensionality', default='low-dimensional')
@click.option('--trials', default=10)
def run_algorithm_comparison_task(dimensionality, trials):
    """
    Function that will run the competing algorithms to Deeplifting.
    The current competitor models are:
    1. IPOPT
    2. Dual Annealing
    3. Differential Evolution
    4. PyGRANSO
    """
    print('Run Algorithms!')
    run = neptune.init_run(  # noqa
        project="dever120/Deeplifting",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzYmIwMTUyNC05YmZmLTQ1NzctOTEyNS1kZTIxYjU5NjY5YjAifQ==",  # noqa
    )  # your credentials
    run['sys/tags'].add(['algorithm-compare', dimensionality])

    print('Run Algorithms!')

    if dimensionality == 'low-dimensional':
        problem_names = low_dimensional_problem_names
        PROBLEMS = PROBLEMS_BY_NAME
    elif dimensionality == 'high-dimensional':
        problem_names = high_dimensional_problem_names
        PROBLEMS = HIGH_DIMENSIONAL_PROBLEMS_BY_NAME
    else:
        raise ValueError('Option for dimensionality does not exist!')

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

        # First run IPOPT
        print('Running IPOPT')
        outputs_ipopt = run_ipopt(problem, trials=trials)

        # Get the final results for all IPOPT runs
        ipopt_results = pd.DataFrame(outputs_ipopt['final_results'], columns=columns)
        ipopt_results['problem_name'] = problem_name
        ipopt_results['hits'] = np.where(
            np.abs(ipopt_results['f'] - minimum_value) <= 1e-4, 1, 0
        )
        ipopt_results['dimensions'] = dimensions

        # Add IPOPT to the problem_performance_list
        problem_performance_list.append(ipopt_results)

        # Next add dual annealing
        print('Running Dual Annealing')
        outputs_dual_annealing = run_dual_annealing(problem, trials=trials)

        # Get the final results for all dual annealing runs
        dual_annleaing_results = pd.DataFrame(
            outputs_dual_annealing['final_results'], columns=columns
        )
        dual_annleaing_results['problem_name'] = problem_name
        dual_annleaing_results['hits'] = np.where(
            np.abs(dual_annleaing_results['f'] - minimum_value) <= 1e-4, 1, 0
        )
        dual_annleaing_results['dimensions'] = dimensions

        # Add dual annealing to the problem_performance_list
        problem_performance_list.append(dual_annleaing_results)

        # Next add differential evolution
        print('Running Differential Evolution')
        outputs_differential_evolution = run_differential_evolution(
            problem, trials=trials
        )

        # Get the final results for all differential evolution runs
        differential_evolution_results = pd.DataFrame(
            outputs_differential_evolution['final_results'], columns=columns
        )
        differential_evolution_results['problem_name'] = problem_name
        differential_evolution_results['hits'] = np.where(
            np.abs(differential_evolution_results['f'] - minimum_value) <= 1e-4, 1, 0
        )
        differential_evolution_results['dimensions'] = dimensions

        # Add differential evolution to the problem_performance_list
        problem_performance_list.append(differential_evolution_results)

        if dimensionality == 'low-dimensional':
            # NOTE: PyGranso takes a lot of power to run locally
            # and was not sustainable on my computer should be run
            # on MSI. Low dimensions should be okay though.

            # Next add pygranso
            print('Running PyGranso!')
            outputs_pygranso = run_pygranso(problem, trials=trials)

            # Get the final results for all differential evolution runs
            pygranso_results = pd.DataFrame(
                outputs_pygranso['final_results'], columns=columns
            )
            pygranso_results['problem_name'] = problem_name
            pygranso_results['hits'] = np.where(
                np.abs(pygranso_results['f'] - minimum_value) <= 1e-4, 1, 0
            )
            pygranso_results['dimensions'] = dimensions

            # Add differential evolution to the problem_performance_list
            problem_performance_list.append(pygranso_results)

        # Next add basin hopping
        print('Running Basin Hopping')
        outputs_bh = run_basinhopping(problem, trials=trials)

        # Get the final results for all basinhoppin runs
        bh_results = pd.DataFrame(outputs_bh['final_results'], columns=columns)
        bh_results['problem_name'] = problem_name
        bh_results['hits'] = np.where(
            np.abs(bh_results['f'] - minimum_value) <= 1e-4, 1, 0
        )
        bh_results['dimensions'] = dimensions

        # Add differential evolution to the problem_performance_list
        problem_performance_list.append(bh_results)

        # Concatenate all of the data at the end of each problem because
        # we can save intermediate results
        problem_performance_df = pd.concat(problem_performance_list, ignore_index=True)
        problem_performance_df['global_minimum'] = minimum_value
        path = f'./algorithm_compare_results/{dimensionality}/{experiment_date}-{problem_name}'  # noqa
        if not os.path.exists(path):
            os.makedirs(path)

        problem_performance_df.to_parquet(f'{path}/{dimensionality}.parquet')


# @cli.command('create-trajectory-plot')
# def run_create_trajectory_plot():
#     """
#     Function that will run each of the models and create a
#     "trajectory plot" for the paper. Every function now has the ability
#     to observe the intermediate trajectory of the optimization with the
#     exception of IPOPT (we need to use a completely different API).
#     With this information we can plot the trajectory of the optimization
#     """
#     print('Create trajectory plot!')
#     # Problem set up
#     problem_name = 'cross_in_tray'
#     trials = 1
#     index = 0
#     problem = PROBLEMS_BY_NAME[problem_name]

#     # First run IPOPT
#     outputs_ipopt = run_ipopt(problem, trials=trials)
#     ipopt_trajectory_data = outputs_ipopt['results'][index, :, :]

#     # For IPOPT we need to manually get the data
#     mask = ~np.isnan(ipopt_trajectory_data).any(axis=1)
#     ipopt_trajectory_data = ipopt_trajectory_data[mask]
#     midpoint = len(ipopt_trajectory_data) // 2
#     ipopt_trajectory_data = ipopt_trajectory_data[[0, midpoint, -1], :2]
#     ipopt_trajectory_data = ipopt_trajectory_data.tolist()

#     # Next add dual annealing
#     outputs_dual_annealing = run_dual_annealing(problem, trials=trials)
#     dual_annealing_trajectory_data = outputs_dual_annealing['callbacks'][
#         index
#     ].x_history

#     # Next add differential evolution
#     outputs_differential_evolution = run_differential_evolution(problem, trials=trials)  # noqa
#     differential_evolution_trajectory_data = outputs_differential_evolution[
#         'callbacks'
#     ][index].x_history

#     # Next add pygranso
#     outputs_pygranso = run_pygranso(problem, trials=trials)
#     pygranso_trajectory_data = outputs_pygranso['callbacks'][index]

#     # Run deeplifting
#     outputs_deeplifting = run_deeplifting(
#         problem, problem_name=problem_name, trials=trials
#     )
#     deeplifting_trajectory_data = outputs_deeplifting['callbacks'][index]

#     # Create models and trajectories
#     trajectories = [
#         deeplifting_trajectory_data.tolist(),
#         ipopt_trajectory_data,
#         dual_annealing_trajectory_data,
#         differential_evolution_trajectory_data,
#         pygranso_trajectory_data,
#     ]

#     models = [
#         'Deeplifting',
#         'IPOPT',
#         'Dual Annealing',
#         'Differential Evolution',
#         'PyGranso',
#     ]

#     # plot the data
#     fig = create_contour_plot(
#         problem_name=problem_name,
#         problem=problem,
#         models=models,
#         trajectories=trajectories,
#         colormap='Greys',
#     )

#     fig.savefig('./images/trajectory.png', bbox_inches='tight', pad_inches=0.05)


# @cli.command('run-deeplifting-and-save')
# def run_saved_model_task():
#     """
#     Run deep lifting over specified available problems and over a search space
#     to find the best performance
#     """
#     hidden_size_128 = (128,)
#     input_sizes = [512]
#     hidden_sizes = [hidden_size_128 * 2]
#     hidden_activations = ['sine']
#     output_activations = ['leaky_relu']
#     agg_functions = ['sum']

#     # Get the available configurations
#     problem_name = 'eggholder'
#     combinations = (
#         input_sizes,
#         hidden_sizes,
#         hidden_activations,
#         output_activations,
#         agg_functions,
#     )
#     configurations = list(product(*combinations))

#     # Number of trials
#     trials = 5

#     # Run over the experiments
#     for (
#         index,
#         (input_size, hidden_size, hidden_activation, output_activation, agg_function),
#     ) in enumerate(configurations):
#         # Get problem_information
#         problem = PROBLEMS_BY_NAME[problem_name]

#         # Get the outputs
#         _ = run_deeplifting(
#             problem,
#             problem_name=problem_name,
#             trials=trials,
#             input_size=input_size,
#             hidden_sizes=hidden_size,
#             activation=hidden_activation,
#             output_activation=output_activation,
#             agg_function=agg_function,
#             save_model_path='./models/',
#         )


# @cli.command('find-best-deeplifting-architecture')
# @click.option('--problem_series', default='ackley')
# @click.option('--method', default='pygranso')
# @click.option('--dimensionality', default='high-dimensional')
# @click.option('--early-stopping', default=False)
# def find_best_architecture_task(problem_series, method, dimensionality, early_stopping):  # noqa
#     """
#     Function that we will use to find the best architecture over multiple
#     "hard" high-dimensional problems. We will aim to tackle a large dimensional
#     space with this function, 2500+
#     """
#     # Set the number of threads to 1
#     os.environ['OMP_NUM_THREADS'] = '1'

#     # Enable wandb
#     wandb.login(key='2080070c4753d0384b073105ed75e1f46669e4bf')

#     wandb.init(
#         # set the wandb project where this run will be logged
#         project="Deeplifting-HD",
#         tags=[f'{method}', f'{problem_series}'],
#     )

#     # Get the problem list
#     if dimensionality == 'high-dimensional':
#         directory = 'high-dimension-search-results'
#         PROBLEMS = HIGH_DIMENSIONAL_PROBLEMS_BY_NAME
#         if problem_series == 'ackley':
#             problem_names = ackley_series
#         elif problem_series == 'alpine1':
#             problem_names = alpine_series
#         elif problem_series == 'chung_reynolds':
#             problem_names = chung_reynolds_series
#         elif problem_series == 'griewank':
#             problem_names = griewank_series
#         elif problem_series == 'lennard_jones':
#             problem_names = lennard_jones_series
#         elif problem_series == 'levy':
#             problem_names = levy_series
#         elif problem_series == 'qing':
#             problem_names = qing_series
#         elif problem_series == 'rastrigin':
#             problem_names = rastrigin_series
#         elif problem_series == 'schwefel':
#             problem_names = schwefel_series
#     elif dimensionality == 'low-dimensional':
#         if problem_series != 'all':
#             raise ValueError('Can only run full list for this option!')
#         directory = 'low-dimension-search-results'
#         PROBLEMS = PROBLEMS_BY_NAME
#         problem_names = low_dimensional_problem_names

#     # Get the available configurations
#     combinations = (
#         search_input_sizes,
#         search_hidden_sizes,
#         search_hidden_activations,
#         search_output_activations,
#         search_agg_functions,
#         search_include_bn,
#     )
#     configurations = list(product(*combinations))
#     trials = 10

#     # List to store performance data
#     performance_df_list = []
#     experiment_date = datetime.today().strftime('%Y-%m-%d-%H')
#     for problem_name in problem_names:
#         # Run over the experiments
#         for (
#             index,
#             (
#                 input_size,
#                 hidden_size,
#                 hidden_activation,
#                 output_activation,
#                 agg_function,
#                 include_bn,
#             ),
#         ) in enumerate(configurations):
#             print(problem_name)

#             # Load the problems
#             problem = PROBLEMS[problem_name]
#             print(
#                 input_size,
#                 hidden_size,
#                 hidden_activation,
#                 output_activation,
#                 agg_function,
#                 include_bn,
#             )

#             # Get the outputs
#             if method == 'pygranso':
#                 if dimensionality == 'high-dimensional':
#                     outputs = run_high_dimensional_deeplifting(
#                         problem,
#                         problem_name=problem_name,
#                         trials=trials,
#                         input_size=input_size,
#                         hidden_sizes=hidden_size,
#                         activation=hidden_activation,
#                         output_activation=output_activation,
#                         agg_function=agg_function,
#                         include_bn=include_bn,
#                     )
#                 elif dimensionality == 'low-dimensional':
#                     outputs = run_deeplifting(
#                         problem,
#                         problem_name=problem_name,
#                         trials=trials,
#                         input_size=input_size,
#                         hidden_sizes=hidden_size,
#                         activation=hidden_activation,
#                         output_activation=output_activation,
#                         agg_function=agg_function,
#                         include_bn=include_bn,
#                         method='single-value',
#                     )

#             elif method == 'pytorch-lbfgs':
#                 outputs = run_lbfgs_deeplifting(
#                     problem,
#                     problem_name=problem_name,
#                     trials=trials,
#                     input_size=input_size,
#                     hidden_sizes=hidden_size,
#                     activation=hidden_activation,
#                     output_activation=output_activation,
#                     agg_function=agg_function,
#                     include_bn=include_bn,
#                 )

#             elif method == 'pytorch-adam':
#                 outputs = run_adam_deeplifting(
#                     problem,
#                     problem_name=problem_name,
#                     trials=trials,
#                     input_size=input_size,
#                     hidden_sizes=hidden_size,
#                     activation=hidden_activation,
#                     output_activation=output_activation,
#                     agg_function=agg_function,
#                     include_bn=include_bn,
#                 )

#             else:
#                 raise ValueError('Method is not supported!')

#             # Get the results of the outputs
#             output_size = problem['dimensions']
#             x_columns = [f'x{i + 1}' for i in range(output_size)]
#             columns = x_columns + ['f', 'f_initial', 'algorithm', 'total_time']

#             results = pd.DataFrame(outputs['final_results'], columns=columns)

#             # Add meta data to the results
#             results['input_size'] = input_size
#             results['hidden_size'] = '-'.join(map(str, hidden_size))
#             results['num_layers'] = len(hidden_size)
#             results['num_neurons'] = hidden_size[0]
#             results['hidden_activation'] = hidden_activation
#             results['output_activation'] = output_activation
#             results['agg_function'] = agg_function
#             results['include_bn'] = include_bn
#             results['problem_name'] = problem_name
#             results['global_minimum'] = problem['global_minimum']
#             results['dimensions'] = output_size
#             results['hits'] = np.abs(results['f'] - results['global_minimum']) <= 1e-4

#             # Print the results
#             hits = results['hits'].mean()
#             run_time = results['total_time'].mean()
#             print(f'Success Rate = {hits}')
#             print(f'Average run time = {run_time}')

#             # Save to parquet
#             layers = len(hidden_size)
#             units = hidden_size[0]

#             path = f'./{directory}/{experiment_date}-{problem_name}'
#             if not os.path.exists(path):
#                 os.makedirs(path)

#             results.to_parquet(
#                 f'{path}/{layers}'
#                 f'-layer-{units}-{agg_function}'
#                 f'-{index}-{method}-{hidden_activation}'
#                 f'-{output_activation}-{include_bn}-'
#                 f'input-size-{input_size}.parquet'  # noqa
#             )

#             # Append performance
#             performance_df_list.append(results)

#             if early_stopping:
#                 if hits >= 0.90:
#                     break


@cli.command('run-pygranso')
@click.option('--problem_series', default='ackley')
@click.option('--dimensionality', default='low-dimensional')
@click.option('--trials', default=10)
def run_pygranso_task(problem_series, dimensionality, trials):
    """
    Function that will run the competing algorithms to Deeplifting.
    The current competitor models are:
    1. PyGRANSO
    """
    print('Run Algorithms!')
    run = neptune.init_run(  # noqa
        project="dever120/Deeplifting",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzYmIwMTUyNC05YmZmLTQ1NzctOTEyNS1kZTIxYjU5NjY5YjAifQ==",  # noqa
    )  # your credentials
    run['sys/tags'].add(['pygranso', dimensionality])

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
        problem_names = low_dimensional_problem_names

    # Create the experiment date
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

        # Next add pygranso
        print('Running PyGranso!')
        outputs_pygranso = run_pygranso(problem, trials=trials)

        # Get the final results for all differential evolution runs
        pygranso_results = pd.DataFrame(
            outputs_pygranso['final_results'], columns=columns
        )
        pygranso_results['problem_name'] = problem_name
        pygranso_results['hits'] = np.where(
            np.abs(pygranso_results['f'] - minimum_value) <= 1e-4, 1, 0
        )
        pygranso_results['dimensions'] = dimensions

        # Add differential evolution to the problem_performance_list
        problem_performance_list.append(pygranso_results)

        # Print the results
        hits = pygranso_results['hits'].mean()
        average_time = pygranso_results['time'].mean()
        print(f'Success Rate = {hits}')
        print(f'Average time = {average_time}')

        # Concatenate all of the data at the end of each problem because
        # we can save intermediate results
        problem_performance_df = pd.concat(problem_performance_list, ignore_index=True)
        path = f'./algorithm_compare_results/{experiment_date}-pygranso-{problem_name}'
        if not os.path.exists(path):
            os.makedirs(path)

        problem_performance_df.to_parquet(f'{path}/{dimensionality}.parquet')


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


# Adding a brand new task but will clean up code
# TODO: Clean up code
@cli.command('find-best-deeplifting-architecture-v2')
@click.option('--problem_name', default='ackley')
@click.option('--method', default='pygranso')
@click.option('--dimensionality', default='low-dimensional')
@click.option('--experimentation', default=True)
def find_best_architecture_task(problem_name, method, dimensionality, experimentation):
    """
    Function that we will use to find the best architecture over multiple
    "hard" high-dimensional problems. We will aim to tackle a large dimensional
    space with this function, 2500+
    """
    # Set the number of threads to 1
    os.environ['OMP_NUM_THREADS'] = '1'

    # Get the available device
    device = get_devices()

    if experimentation:
        # Enable wandb
        wandb.login(key='2080070c4753d0384b073105ed75e1f46669e4bf')

        wandb.init(
            # set the wandb project where this run will be logged
            project="Deeplifting-HD",
            tags=[f'{method}', f'{problem_name}'],
        )

    # Setup the problem
    if dimensionality == 'low-dimensional':
        directory = 'low-dimension'
        PROBLEMS = PROBLEMS_BY_NAME

    elif dimensionality == 'high-dimensional':
        directory = 'high-dimension'
        PROBLEMS = HIGH_DIMENSIONAL_PROBLEMS_BY_NAME

    else:
        raise ValueError(f'{dimensionality} is not valid!')
    problem = PROBLEMS[problem_name]

    # Objective function
    objective = problem['objective']

    # Bounds
    bounds = problem['bounds']

    # Get the device (CPU for now)
    output_size = problem['dimensions']

    # Setup list to store information
    objective_values = []
    initial_values = []
    indexes = []
    network_config = []
    results_df_list = []
    run_times = []
    iterations = []
    fn_evals = []

    # Layer search
    minimum_num_layers = 2
    maximum_num_layers = 10

    # Number of neurons
    units_search = [64, 128, 256]

    # Initial layer type
    input_dimension = 64
    initial_layer_type = 'linear'
    include_weight_initialization = True
    include_bn = True

    # Maximum number of trials
    # Let's actually try and do 50 - we will use the linear
    # method
    trials = 50

    for num_layers in reversed(range(minimum_num_layers, maximum_num_layers + 1)):
        for units in units_search:
            # n-layer m neuron network
            hidden_sizes = (units,) * num_layers

            # We have an observation that we can start at the same point
            # but it may or may not converge so we can try different
            # weights
            for index, trial in enumerate(range(trials)):
                # Fix the inputs for deeplifting
                if initial_layer_type == 'embedding':
                    inputs = torch.randint(
                        low=0, high=(units - 1), size=(input_dimension, units)
                    )
                    inputs = inputs.to(device=device, dtype=torch.long)

                elif initial_layer_type == 'linear':
                    inputs = torch.randn(size=(input_dimension, units))
                    inputs = inputs.to(device=device, dtype=torch.double)

                else:
                    raise ValueError(f'{initial_layer_type} is not an accepted type!')

                # Initialization for other models
                x_start = initialize_vector(size=output_size, bounds=bounds)
                x_start = torch.tensor(x_start)
                x_start = x_start.to(device=device, dtype=torch.double)

                # Creates different weight intializations for the same starting point
                # x0
                for i in range(10, 110, 10):
                    print(f'Fitting point {x_start} - with weights {i}')
                    print(
                        f' - layers - {num_layers} - units - {units} - trial - {trial}'
                    )

                    # Deeplifting model with skip connections
                    model = ReLUDeepliftingMLP(
                        initial_hidden_size=units,
                        hidden_sizes=hidden_sizes,
                        output_size=output_size,
                        bounds=bounds,
                        initial_layer_type=initial_layer_type,
                        include_weight_initialization=include_weight_initialization,
                        include_bn=include_bn,
                        seed=i,
                    )

                    model = model.to(device=device, dtype=torch.double)

                    # Let's make sure that all methods have the same x0
                    # Test
                    train_model_to_output(
                        inputs=inputs,
                        model=model,
                        x0=x_start,
                        epochs=5000,
                        lr=1,
                        tolerance=1e-3,
                    )
                    nvar = getNvarTorch(model.parameters())

                    # Setup a pygransoStruct for the algorithm
                    # options
                    opts = pygransoStruct()

                    # Print the model outputs and check against x0 also
                    # want to use a print out to make sure all models have
                    # the same starting point
                    model.eval()
                    outputs = model(inputs=inputs)

                    # Inital x0 - for neural network
                    x0 = (
                        torch.nn.utils.parameters_to_vector(model.parameters())
                        .detach()
                        .reshape(nvar, 1)
                        .to(device=device, dtype=torch.double)
                    )

                    # PyGranso options
                    opts.x0 = x0
                    opts.torch_device = device
                    opts.print_frequency = 1
                    opts.limited_mem_size = 100
                    opts.stat_l2_model = False
                    opts.double_precision = True

                    # opts.disable_terminationcode_6 = True
                    # opts.halt_on_linesearch_bracket = False

                    opts.opt_tol = 1e-10
                    opts.maxit = 1000

                    # TODO: Clean up meaningless variables
                    results = None
                    deeplifting_results = None

                    # Set up the function with the results
                    fn = lambda x: objective(  # noqa
                        x, results=results, trial=trial, version='pytorch'
                    )

                    # Get the objective value at the initial point
                    outputs = model(inputs=inputs)

                    # Get the initial value of the objective
                    init_fn = lambda x: objective(  # noqa
                        x, results=None, trial=0, version='pytorch'
                    )
                    x_init, f_init = deeplifting_predictions(outputs, init_fn)
                    f_init = f_init.detach().cpu().numpy()

                    # Combined function
                    comb_fn = lambda model: deeplifting_nd_fn(  # noqa
                        model,
                        fn,
                        trial,
                        output_size,
                        deeplifting_results,
                        inputs=inputs,
                    )  # noqa

                    # Run the main algorithm
                    model.train()
                    start = time.time()
                    soln = pygranso(var_spec=model, combined_fn=comb_fn, user_opts=opts)
                    end = time.time()
                    total_time = end - start

                    # Hits
                    hit = int(
                        np.abs(problem['global_minimum'] - soln.best.f)
                        / np.abs(f_init - problem['global_minimum'])
                        <= 1e-4
                    )
                    objective_values.append(
                        (hit, f_init, problem['global_minimum'], soln.best.f)
                    )
                    initial_values.append(x_start.detach().cpu().numpy())
                    indexes.append(index)
                    network_config.append((num_layers, units))
                    run_times.append(total_time)
                    iterations.append(soln.iters)
                    fn_evals.append(soln.fn_evals)

                    if hit == 1.0:
                        break

            # Create the data from this run and save sequentially
            # For the starting positions let's save as a json string
            columns = [f'x{i + 1}' for i in range(output_size)]
            xs = json.dumps(dict(zip(columns, initial_values[0])))

            # Initialize results
            results_df = pd.DataFrame(
                np.asarray(objective_values),
                columns=['success', 'f_init', 'global_minimum', 'f'],
            )
            results_df[['num_layers', 'units']] = network_config
            results_df['index'] = indexes
            results_df['total_time'] = run_times
            results_df['xs'] = xs

            # Save the results
            results_df_list.append(results_df)
            main_directory = '/home/jusun/dever120/Deeplifting'
            deeplifting_directory = (
                'experiments/3b39b4fb-0520-4795-aaba-a8eab24ff8fd/'
                f'{directory}/deeplifting-pygranso'
            )
            filename = os.path.join(
                main_directory,
                deeplifting_directory,
                f'{problem_name}-relu-{num_layers}-{units}.parquet',
            )
            results_df.to_parquet(filename)


if __name__ == "__main__":
    # Be able to run different commands
    cli()
