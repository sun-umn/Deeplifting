#!/usr/bin/python
# stdlib
from itertools import product

# third party
import click
import neptune
import numpy as np
import pandas as pd

# first party
from deeplifting.optimization import (
    run_deeplifting,
    run_differential_evolution,
    run_dual_annealing,
    run_ipopt,
    run_pygranso,
)
from deeplifting.problems import PROBLEMS_BY_NAME
from deeplifting.utils import create_contour_plot

# Identify problems to run
low_dimensional_problem_names = [
    # 'ackley',
    # 'bukin_n6',
    # 'cross_in_tray',
    # 'drop_wave',
    # 'eggholder',
    # 'griewank',
    # 'holder_table',
    # 'levy',
    # 'levy_n13',
    # 'rastrigin',
    # 'schaffer_n2',
    # 'schaffer_n4',
    # 'schwefel',
    # 'shubert',
    # 'ex8_1_1',
    # 'kriging_peaks_red010',
    # 'kriging_peaks_red020',
    # 'kriging_peaks_red030',
    # 'mathopt6',
    # 'quantum',
    # 'rosenbrock',
    # 'cross_leg_table',
    'sine_envelope',
    'ackley2',
    'ackley3',
    # 'ackley4',  # Was having issues
    'adjiman',
    'alpine1',
    'alpine2',
    'bartels_conn',
    'beale',
    'biggs_exp2',
    'bird',
    'bohachevsky1',
    'bohachevsky2',
    'bohachevsky3',
    'booth',
    'branin_rcos',
    'brent',
    'brown',
    'bukin_n2',
    'bukin_n4',
    'camel_3hump',
    'camel_6hump',
    'chen_bird',
    'chen_v',
    'chichinadze',
    'chung_reynolds',
    'cube',
]

high_dimensional_problem_names = [
    'ackley_30d',
    'ackley_100d',
    'ackley_1000d',
    'ex8_6_2',
]

# Identify available hidden sizes
hidden_size_64 = (64,)
hidden_size_128 = (128,)
hidden_size_256 = (256,)
hidden_size_512 = (512,)
hidden_size_768 = (768,)
hidden_size_1024 = (1024,)
hidden_size_2048 = (2048,)

# Hidden size combinations
hidden_sizes = [
    # Hidden sizes of 128
    hidden_size_128 * 2,
    # hidden_size_128 * 3,
    # hidden_size_128 * 4,
    # hidden_size_128 * 5,
    # # Hidden sizes of 256
    # hidden_size_256 * 2,
    # hidden_size_256 * 3,
    # hidden_size_256 * 4,
    # hidden_size_256 * 5,
    # # Hidden sizes of 512
    # hidden_size_512 * 2,
    # hidden_size_512 * 3,
    # hidden_size_512 * 4,
    # hidden_size_512 * 5,
    # # Hidden sizes of 768
    # hidden_size_768 * 2,
    # hidden_size_768 * 3,
    # hidden_size_768 * 4,
    # hidden_size_768 * 5,
    # # Hidden sizes of 1024
    # hidden_size_1024 * 2,
    # hidden_size_1024 * 3,
    # hidden_size_1024 * 4,
    # # Hidden sizes of 2048
    # hidden_size_2048 * 2,
    # hidden_size_2048 * 3,
    # hidden_size_2048 * 4,
]

# Input sizes
input_sizes = [512]

# Hidden activations
hidden_activations = ['sine']

# Ouput activations
output_activations = ['leaky_relu']

# Aggregate functions - for skip connections
agg_functions = ['sum']


@click.group()
def cli():
    pass


@cli.command('run-deeplifting-task')
@click.option('--dimensionality', default='low-dimensional')
def run_deeplifting_task(dimensionality):
    """
    Run deep lifting over specified available problems and over a search space
    to find the best performance
    """
    # Enable the neptune run
    # Get api token
    # TODO: If api token is not present log a warning
    # and default to saving files locally
    run = neptune.init_run(  # noqa
        project="dever120/Deeplifting",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzYmIwMTUyNC05YmZmLTQ1NzctOTEyNS1kZTIxYjU5NjY5YjAifQ==",  # noqa
    )  # your credentials

    if dimensionality == 'low-dimensional':
        problem_names = low_dimensional_problem_names
    elif dimensionality == 'high-dimensional':
        problem_names = high_dimensional_problem_names
    else:
        raise ValueError('Option for dimensionality does not exist!')

    # Get the available configurations
    combinations = (
        input_sizes,
        hidden_sizes,
        hidden_activations,
        output_activations,
        agg_functions,
    )
    configurations = list(product(*combinations))

    # Number of trials
    trials = 20

    # List to store performance data
    performance_df_list = []

    # Run over the experiments
    for index, (
        input_size,
        hidden_size,
        hidden_activation,
        output_activation,
        agg_function,
    ) in enumerate(configurations):
        for problem_name in problem_names:
            # Load the problems
            problem = PROBLEMS_BY_NAME[problem_name]

            # Get the outputs
            outputs = run_deeplifting(
                problem,
                problem_name=problem_name,
                trials=trials,
                input_size=input_size,
                hidden_sizes=hidden_size,
                activation=hidden_activation,
                output_activation=output_activation,
                agg_function=agg_function,
            )

            # Get the results of the outputs
            output_size = problem['dimensions']
            x_columns = [f'x{i + 1}' for i in range(output_size)]
            columns = x_columns + ['f', 'algorithm', 'total_time']

            results = pd.DataFrame(
                outputs['final_results'],
                columns=columns,
            )

            # Add meta data to the results
            results['input_size'] = input_size
            results['hidden_size'] = '-'.join(map(str, hidden_sizes))
            results['hidden_activation'] = hidden_activation
            results['output_activation'] = output_activation
            results['agg_function'] = agg_function
            results['problem_name'] = problem_name
            results['global_minimum'] = problem['global_minimum']

            # Save to parquet
            results.to_parquet(
                f'./results/results-2023-08-03-2-{problem_name}-{index}.parquet'
            )

            # Append performance
            performance_df_list.append(results)


@cli.command('run-algorithm-comparisons')
@click.option('--dimensionality', default='low-dimensional')
def run_algorithm_comparison_task(dimensionality):
    """
    Function that will run the competing algorithms to Deeplifting.
    The current competitor models are:
    1. IPOPT
    2. Dual Annealing
    3. Differential Evolution
    4. PyGRANSO
    """
    print('Run Algorithms!')
    trials = 20

    if dimensionality == 'low-dimensional':
        problem_names = low_dimensional_problem_names
    elif dimensionality == 'high-dimensional':
        problem_names = high_dimensional_problem_names
    else:
        raise ValueError('Option for dimensionality does not exist!')

    for problem_name in problem_names:
        problem_performance_list = []

        # Setup the problem
        problem = PROBLEMS_BY_NAME[problem_name]

        # Get the known minimum
        minimum_value = problem['global_minimum']

        # First run IPOPT
        outputs_ipopt = run_ipopt(problem, trials=trials)

        # Get the final results for all IPOPT runs
        ipopt_results = pd.DataFrame(
            outputs_ipopt['final_results'],
            columns=['x1', 'x2', 'f', 'algorithm', 'time'],
        )
        ipopt_results['problem_name'] = problem_name
        ipopt_results['hits'] = np.where(
            np.abs(ipopt_results['f'] - minimum_value) <= 1e-4, 1, 0
        )

        # Add IPOPT to the problem_performance_list
        problem_performance_list.append(ipopt_results)

        # Next add dual annealing
        outputs_dual_annealing = run_dual_annealing(problem, trials=trials)

        # Get the final results for all dual annealing runs
        dual_annleaing_results = pd.DataFrame(
            outputs_dual_annealing['final_results'],
            columns=['x1', 'x2', 'f', 'algorithm', 'time'],
        )
        dual_annleaing_results['problem_name'] = problem_name
        dual_annleaing_results['hits'] = np.where(
            np.abs(dual_annleaing_results['f'] - minimum_value) <= 1e-4, 1, 0
        )

        # Add dual annealing to the problem_performance_list
        problem_performance_list.append(dual_annleaing_results)

        # Next add differential evolution
        outputs_differential_evolution = run_differential_evolution(
            problem, trials=trials
        )

        # Get the final results for all differential evolution runs
        differential_evolution_results = pd.DataFrame(
            outputs_differential_evolution['final_results'],
            columns=['x1', 'x2', 'f', 'algorithm', 'time'],
        )
        differential_evolution_results['problem_name'] = problem_name
        differential_evolution_results['hits'] = np.where(
            np.abs(differential_evolution_results['f'] - minimum_value) <= 1e-4, 1, 0
        )

        # Add differential evolution to the problem_performance_list
        problem_performance_list.append(differential_evolution_results)

        # Next add pygranso
        outputs_pygranso = run_pygranso(problem, trials=trials)

        # Get the final results for all differential evolution runs
        pygranso_results = pd.DataFrame(
            outputs_pygranso['final_results'],
            columns=['x1', 'x2', 'f', 'algorithm', 'time'],
        )
        pygranso_results['problem_name'] = problem_name
        pygranso_results['hits'] = np.where(
            np.abs(pygranso_results['f'] - minimum_value) <= 1e-4, 1, 0
        )

        # Add differential evolution to the problem_performance_list
        problem_performance_list.append(pygranso_results)

        # Concatenate all of the data at the end of each problem because
        # we can save intermediate results
        problem_performance_df = pd.concat(problem_performance_list, ignore_index=True)
        problem_performance_df.to_parquet(
            f'./results/algorithm-comparisons-{problem_name}.parquet'
        )


@cli.command('create-trajectory-plot')
def run_create_trajectory_plot():
    """
    Function that will run each of the models and create a
    "trajectory plot" for the paper. Every function now has the ability
    to observe the intermediate trajectory of the optimization with the
    exception of IPOPT (we need to use a completely different API).
    With this information we can plot the trajectory of the optimization
    """
    print('Create trajectory plot!')
    # Problem set up
    problem_name = 'eggholder'
    trials = 1
    index = 0
    problem = PROBLEMS_BY_NAME[problem_name]

    # First run IPOPT
    outputs_ipopt = run_ipopt(problem, trials=trials)
    ipopt_trajectory_data = outputs_ipopt['results'][index, :, :]

    # For IPOPT we need to manually get the data
    mask = ~np.isnan(ipopt_trajectory_data).any(axis=1)
    ipopt_trajectory_data = ipopt_trajectory_data[mask]
    midpoint = len(ipopt_trajectory_data) // 2
    ipopt_trajectory_data = ipopt_trajectory_data[[0, midpoint, -1], :2]
    ipopt_trajectory_data = ipopt_trajectory_data.tolist()

    # Next add dual annealing
    outputs_dual_annealing = run_dual_annealing(problem, trials=trials)
    dual_annealing_trajectory_data = outputs_dual_annealing['callbacks'][
        index
    ].x_history

    # Next add differential evolution
    outputs_differential_evolution = run_differential_evolution(problem, trials=trials)
    differential_evolution_trajectory_data = outputs_differential_evolution[
        'callbacks'
    ][index].x_history

    # Next add pygranso
    outputs_pygranso = run_pygranso(problem, trials=trials)
    pygranso_trajectory_data = outputs_pygranso['callbacks'][index]

    # Run deeplifting
    outputs_deeplifting = run_deeplifting(
        problem, problem_name=problem_name, trials=trials
    )
    deeplifting_trajectory_data = outputs_deeplifting['callbacks'][index]

    # Create models and trajectories
    trajectories = [
        deeplifting_trajectory_data,
        ipopt_trajectory_data,
        dual_annealing_trajectory_data,
        differential_evolution_trajectory_data,
        pygranso_trajectory_data,
    ]

    models = [
        'Deeplifting',
        'IPOPT',
        'Dual Annealing',
        'Differential Evolution',
        'PyGranso',
    ]

    # plot the data
    fig = create_contour_plot(
        problem_name=problem_name,
        problem=problem,
        models=models,
        trajectories=trajectories,
        colormap='OrRd_r',
    )

    return fig


@cli.command('run-deeplifting-and-save')
def run_saved_model_task():
    """
    Run deep lifting over specified available problems and over a search space
    to find the best performance
    """
    # Get the available configurations
    problem_name = 'eggholder'
    combinations = (
        input_sizes,
        hidden_sizes,
        hidden_activations,
        output_activations,
        agg_functions,
    )
    configurations = list(product(*combinations))

    # Number of trials
    trials = 5

    # Run over the experiments
    for index, (
        input_size,
        hidden_size,
        hidden_activation,
        output_activation,
        agg_function,
    ) in enumerate(configurations):
        # Get problem_information
        problem = PROBLEMS_BY_NAME[problem_name]

        # Get the outputs
        _ = run_deeplifting(
            problem,
            problem_name=problem_name,
            trials=trials,
            input_size=input_size,
            hidden_sizes=hidden_size,
            activation=hidden_activation,
            output_activation=output_activation,
            agg_function=agg_function,
            save_model_path='./models/',
        )


if __name__ == "__main__":
    # Be able to run different commands
    cli()
