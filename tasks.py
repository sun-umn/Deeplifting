#!/usr/bin/python
# stdlib
import os
from datetime import datetime
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
    run_high_dimensional_deeplifting,
    run_ipopt,
    run_lbfgs_deeplifting,
    run_pygranso,
    run_pyomo,
)
from deeplifting.problems import HIGH_DIMENSIONAL_PROBLEMS_BY_NAME, PROBLEMS_BY_NAME
from deeplifting.utils import create_contour_plot

# Identify problems to run
low_dimensional_problem_names = [
    'ackley',  # Low
    'bukin_n6',  # High, 2 layer is best so far, takes a while to run
    'cross_in_tray',  # Low, runs quickly
    'drop_wave',  # Low, runs quickly
    'eggholder',  # Medium, takes time to run
    'griewank',  # Low, (1.0 with 3-layer, 0.95 2-layer)
    'holder_table',  # Medium
    'levy',  # Low, 3-layer
    'levy_n13',  # Low, 3-layer
    'rastrigin',  # Low, 3-layer
    'schaffer_n2',  # Low, 3-layer
    'schaffer_n4',  # Low, 3-layer
    'schwefel',  # Takes a while to run, DA is better at 100% but we are at 85%
    # 'shubert',  # Takes a while to run,
    # 'ex8_1_1',
    # 'kriging_peaks_red010',
    # 'kriging_peaks_red020',
    # 'mathopt6',
    # 'quantum',
    # 'rosenbrock',
    # 'cross_leg_table',
    # 'sine_envelope',  # Having issues
    # 'ackley2',
    # 'ackley3',  # Having issues
    # 'ackley4',  # Having issues
    # 'adjiman',
    # 'alpine1',
    # 'alpine2',
    # 'bartels_conn',
    # 'beale',
    # # 'biggs_exp2',  # Having issues
    # 'bird',
    # 'bohachevsky1',
    # 'bohachevsky2',  # Having issues
    # 'bohachevsky3',
    # 'booth',
    # 'branin_rcos',  # Having issues
    # 'brent',
    # 'brown',
    # 'bukin_n2',
    # 'bukin_n4',
    # 'camel_3hump',
    # 'camel_6hump',
    # 'chen_bird',  # Having issues
    # 'chen_v',  # Having issues
    # 'chichinadze',  # Having issues
    # 'chung_reynolds',
    # 'cube',  # Correct but paper has wrong x
    # 'xinsheyang_n2',
    # 'xinsheyang_n3',
    # 'layeb12',
    # 'layeb3',
    # 'layeb4',
    # 'layeb6',
    # 'layeb7',
    # 'layeb8',
]

high_dimensional_problem_names = [
    # Ackley Series - Origin Solution
    'ackley_3d',
    'ackley_5d',
    'ackley_30d',
    'ackley_100d',
    'ackley_500d',
    'ackley_1000d',
    # # Alpine1 Series - Origin Solution
    'alpine1_3d',
    'alpine1_5d',
    'alpine1_30d',
    'alpine1_100d',
    'alpine1_500d',
    'alpine1_1000d',
    # Chung-Reynolds Series - Origin Solution
    'chung_reyonlds_3d',
    'chung_reynolds_5d',
    'chung_reynolds_30d',
    'chung_reynolds_100d',
    'chung_reynolds_500d',
    'chung_reynolds_1000d',
    # Griewank Series - Origin Solution
    'griewank_3d',
    'griewank_5d',
    'griewank_30d',
    'griewank_100d',
    'griewank_500d',
    'griewank_1000d',
    # Layeb 4 Series - Non-origin solution
    'layeb4_3d',
    'layeb4_5d',
    'layeb4_30d',
    'layeb4_100d',
    'layeb4_500d',
    'layeb4_1000d',
    # Levy Series - Non-origin solution
    'levy_3d',
    'levy_5d',
    'levy_30d',
    'levy_100d',
    'levy_500d',
    'levy_1000d',
    # Qing Series - Non-origin solution
    'qing_3d',
    'qing_5d',
    'qing_30d',
    'qing_100d',
    'qing_500d',
    'qing_1000d',
    # Rastrigin series - Origin solution
    'rastrigin_3d',
    'rastrigin_5d',
    'rastrigin_30d',
    'rastrigin_100d',
    'rastrigin_500d',
    'rastrigin_1000d',
    # Schewefel series - Non-origin solution
    'schwefel_3d',
    'schwefel_5d',
    'schwefel_30d',
    'schwefel_100d',
    'schwefel_500d',
    'schwefel_1000d',
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
search_hidden_sizes = [
    # Hidden sizes of 128
    hidden_size_128 * 2,
    hidden_size_128 * 3,
    hidden_size_128 * 4,
    hidden_size_128 * 5,
    # Hidden sizes of 256
    hidden_size_256 * 2,
    hidden_size_256 * 3,
    hidden_size_256 * 4,
    hidden_size_256 * 5,
    # Hidden sizes of 512
    hidden_size_512 * 2,
    hidden_size_512 * 3,
    hidden_size_512 * 4,
    hidden_size_512 * 5,
    # Try an Autoencode Architecture
    (256, 128, 256),
    (512, 128, 512),
]

# Input sizes
search_input_sizes = [128, 512]

# Hidden activations
search_hidden_activations = ['sine', 'relu', 'leaky_relu']

# Ouput activations
search_output_activations = ['leaky_relu', 'sine']

# Aggregate functions - for skip connections
search_agg_functions = ['sum', 'max']


@click.group()
def cli():
    pass


@cli.command('run-deeplifting-task')
@click.option('--dimensionality', default='low-dimensional')
@click.option('--layers', default=2)
@click.option('--units', default=128)
@click.option('--method', default='particle')
@click.option('--output_activation', default='leaky_relu')
@click.option('--agg_function', default='sum')
@click.option('--trials', default=20)
def run_deeplifting_task(
    dimensionality, layers, method, output_activation, units, agg_function, trials
):
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

    input_sizes = [512]
    hidden_activations = ['sine']

    if dimensionality == 'low-dimensional':
        problem_names = low_dimensional_problem_names
        PROBLEMS = PROBLEMS_BY_NAME
    elif dimensionality == 'high-dimensional':
        problem_names = high_dimensional_problem_names
        PROBLEMS = HIGH_DIMENSIONAL_PROBLEMS_BY_NAME
    else:
        raise ValueError('Option for dimensionality does not exist!')

    # Configuarable number of units / neurons
    if units == 128:
        hidden_size_units = hidden_size_128
    elif units == 512:
        hidden_size_units = hidden_size_512
    else:
        raise ValueError(f'{units} units is not supported')

    # Configurable number of layers
    if layers == 2:
        dl_hidden_sizes = [hidden_size_units * 2]
    elif layers == 3:
        dl_hidden_sizes = [hidden_size_units * 3]
    elif layers == 4:
        dl_hidden_sizes = [hidden_size_units * 4]
    else:
        raise ValueError('This many layers is not yet configured!')

    # Configurable output activation function
    if output_activation == 'sine':
        output_activations = ['sine']
    elif output_activation == 'leaky_relu':
        output_activations = ['leaky_relu']
    else:
        raise ValueError(f'{output_activation} not supported!')

    # Aggregate function
    if agg_function == 'sum':
        agg_functions = ['sum']
    elif agg_function == 'max':
        agg_functions = ['max']

    # Get the available configurations
    combinations = (
        input_sizes,
        dl_hidden_sizes,
        hidden_activations,
        output_activations,
        agg_functions,
    )
    configurations = list(product(*combinations))

    # List to store performance data
    performance_df_list = []

    # Run over the experiments
    for (
        index,
        (input_size, hidden_size, hidden_activation, output_activation, agg_function),
    ) in enumerate(configurations):
        for problem_name in problem_names:
            print(problem_name)
            # Load the problems
            problem = PROBLEMS[problem_name]

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
                method=method,
            )

            # Get the results of the outputs
            output_size = problem['dimensions']
            x_columns = [f'x{i + 1}' for i in range(output_size)]
            columns = x_columns + ['f', 'algorithm', 'total_time']

            results = pd.DataFrame(outputs['final_results'], columns=columns)

            # Add meta data to the results
            results['input_size'] = input_size
            results['hidden_size'] = '-'.join(map(str, hidden_size))
            results['hidden_activation'] = hidden_activation
            results['output_activation'] = output_activation
            results['agg_function'] = agg_function
            results['problem_name'] = problem_name
            results['global_minimum'] = problem['global_minimum']
            results['dimensions'] = output_size

            # Save to parquet
            results.to_parquet(
                f'./results/results-2023-08-{layers}-layer-{units}-{agg_function}'
                f'-{problem_name}-{index}-{method}-{output_activation}'
                f'{dimensionality}.parquet'  # noqa
            )

            # Append performance
            performance_df_list.append(results)


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

    if dimensionality == 'low-dimensional':
        problem_names = low_dimensional_problem_names
        PROBLEMS = PROBLEMS_BY_NAME
    elif dimensionality == 'high-dimensional':
        problem_names = high_dimensional_problem_names
        PROBLEMS = HIGH_DIMENSIONAL_PROBLEMS_BY_NAME
    else:
        raise ValueError('Option for dimensionality does not exist!')

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

        # # Next add pygranso
        # print('Running PyGranso!')
        # outputs_pygranso = run_pygranso(problem, trials=trials)

        # # Get the final results for all differential evolution runs
        # pygranso_results = pd.DataFrame(
        #     outputs_pygranso['final_results'], columns=columns
        # )
        # pygranso_results['problem_name'] = problem_name
        # pygranso_results['hits'] = np.where(
        #     np.abs(pygranso_results['f'] - minimum_value) <= 1e-4, 1, 0
        # )
        # pygranso_results['dimensions'] = dimensions

        # # Add differential evolution to the problem_performance_list
        # problem_performance_list.append(pygranso_results)

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

        # Add differential evolution to the problem_performance_list
        problem_performance_list.append(scip_results)

        # Concatenate all of the data at the end of each problem because
        # we can save intermediate results
        problem_performance_df = pd.concat(problem_performance_list, ignore_index=True)
        experiment_date = datetime.today().strftime('%Y-%m-%d-%H-%m')
        path = f'./algorithm_compare_results/{experiment_date}-{problem_name}'
        if not os.path.exists(path):
            os.makedirs(path)

        problem_performance_df.to_parquet(f'{path}/{dimensionality}.parquet')


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
    problem_name = 'cross_in_tray'
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
        deeplifting_trajectory_data.tolist(),
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
        colormap='Greys',
    )

    fig.savefig('./images/trajectory.png', bbox_inches='tight', pad_inches=0.05)


@cli.command('run-deeplifting-and-save')
def run_saved_model_task():
    """
    Run deep lifting over specified available problems and over a search space
    to find the best performance
    """
    input_sizes = [512]
    hidden_sizes = [hidden_size_128 * 2]
    hidden_activations = ['sine']
    output_activations = ['leaky_relu']
    agg_functions = ['sum']

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
    for (
        index,
        (input_size, hidden_size, hidden_activation, output_activation, agg_function),
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


@cli.command('find-best-deeplifting-architecture')
@click.option('--problem_name', default='ackley_2500d')
@click.option('--method', default='pygranso')
@click.option('--dimensionality', default='high-dimensional')
def find_best_architecture_task(problem_name, method, dimensionality):
    """
    Function that we will use to find the best architecture over multiple
    "hard" high-dimensional problems. We will aim to tackle a large dimensional
    space with this function, 2500+
    """
    # Enable the neptune run
    # Get api token
    # TODO: If api token is not present log a warning
    # and default to saving files locally
    run = neptune.init_run(  # noqa
        project="dever120/Deeplifting",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzYmIwMTUyNC05YmZmLTQ1NzctOTEyNS1kZTIxYjU5NjY5YjAifQ==",  # noqa
    )  # your credentials
    run['sys/tags'].add([problem_name, method])

    # Get the problem list
    if dimensionality == 'high-dimensional':
        PROBLEMS = HIGH_DIMENSIONAL_PROBLEMS_BY_NAME
    elif dimensionality == 'low-dimensional':
        PROBLEMS = PROBLEMS_BY_NAME

    # Get the available configurations
    combinations = (
        search_input_sizes,
        search_hidden_sizes,
        search_hidden_activations,
        search_output_activations,
        search_agg_functions,
    )
    configurations = list(product(*combinations))
    trials = 10

    # List to store performance data
    performance_df_list = []

    # Run over the experiments
    for (
        index,
        (input_size, hidden_size, hidden_activation, output_activation, agg_function),
    ) in enumerate(configurations):
        print(problem_name)
        # Load the problems
        problem = PROBLEMS[problem_name]

        # Get the outputs
        if method == 'pygranso':
            if dimensionality == 'high-dimensional':
                outputs = run_high_dimensional_deeplifting(
                    problem,
                    problem_name=problem_name,
                    trials=trials,
                    input_size=input_size,
                    hidden_sizes=hidden_size,
                    activation=hidden_activation,
                    output_activation=output_activation,
                    agg_function=agg_function,
                )
            elif dimensionality == 'low-dimensional':
                outputs = run_deeplifting(
                    problem,
                    problem_name=problem_name,
                    trials=trials,
                    input_size=input_size,
                    hidden_sizes=hidden_size,
                    activation=hidden_activation,
                    output_activation=output_activation,
                    agg_function=agg_function,
                    method='single-value',
                )

        elif method == 'pytorch-lbfgs':
            outputs = run_lbfgs_deeplifting(
                problem,
                problem_name=problem_name,
                trials=trials,
                input_size=input_size,
                hidden_sizes=hidden_size,
                activation=hidden_activation,
                output_activation=output_activation,
                agg_function=agg_function,
            )
        else:
            raise ValueError('Method is not supported!')

        # Get the results of the outputs
        output_size = problem['dimensions']
        x_columns = [f'x{i + 1}' for i in range(output_size)]
        columns = x_columns + ['f', 'algorithm', 'total_time']

        results = pd.DataFrame(outputs['final_results'], columns=columns)

        # Add meta data to the results
        results['input_size'] = input_size
        results['hidden_size'] = '-'.join(map(str, hidden_size))
        results['hidden_activation'] = hidden_activation
        results['output_activation'] = output_activation
        results['agg_function'] = agg_function
        results['problem_name'] = problem_name
        results['global_minimum'] = problem['global_minimum']
        results['dimensions'] = output_size

        # Save to parquet
        layers = len(hidden_size)
        units = hidden_size[0]

        experiment_date = datetime.today().strftime('%Y-%m-%d-%H-%m')
        path = f'./search_results/{experiment_date}-{problem_name}'
        if not os.path.exists(path):
            os.makedirs(path)

        results.to_parquet(
            f'{path}/{dimensionality}-{layers}'
            f'-layer-{units}-{agg_function}'
            f'-{index}-{method}-{output_activation}-'
            f'input-size-{input_size}.parquet'  # noqa
        )

        # Append performance
        performance_df_list.append(results)


@cli.command('run-pygranso')
@click.option('--dimensionality', default='low-dimensional')
@click.option('--trials', default=10)
def run_pygranso_task(dimensionality, trials):
    """
    Function that will run the competing algorithms to Deeplifting.
    The current competitor models are:
    1. PyGRANSO
    """
    print('Run Algorithms!')

    if dimensionality == 'low-dimensional':
        problem_names = low_dimensional_problem_names
        PROBLEMS = PROBLEMS_BY_NAME
    elif dimensionality == 'high-dimensional':
        problem_names = high_dimensional_problem_names
        PROBLEMS = HIGH_DIMENSIONAL_PROBLEMS_BY_NAME
    else:
        raise ValueError('Option for dimensionality does not exist!')

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

        # Concatenate all of the data at the end of each problem because
        # we can save intermediate results
        problem_performance_df = pd.concat(problem_performance_list, ignore_index=True)
        path = f'./algorithm_compare_results/{experiment_date}-pygranso-{problem_name}'
        if not os.path.exists(path):
            os.makedirs(path)

        problem_performance_df.to_parquet(f'{path}/{dimensionality}.parquet')


if __name__ == "__main__":
    # Be able to run different commands
    cli()
