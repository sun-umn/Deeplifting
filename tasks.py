#!/usr/bin/python
# stdlib
from itertools import product

# third party
import click
import neptune
import pandas as pd

# first party
from deeplifting.optimization import run_deeplifting
from deeplifting.problems import PROBLEMS_BY_NAME

# Identify problems to run
problem_names = [
    'ackley',
    'bukin_n6',
    'cross_in_tray',
    'drop_wave',
    'eggholder',
    'griewank',
    'holder_table',
    'levy',
    'levy_n13',
    'rastrigin',
    'schaffer_n2',
    'schaffer_n4',
    'schwefel',
    'shubert',
    'ex8_1_1',
    'kriging_peaks_red010',
    'kriging_peaks_red020',
    'mathopt6',
    'quantum',
    'rosenbrock',
    'cross_leg_table',
    'sine_envelope',
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
def run_deeplifting_task():
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
                trials=trials,
                input_size=input_size,
                hidden_sizes=hidden_size,
                activation=hidden_activation,
                output_activation=output_activation,
                agg_function=agg_function,
            )

            # Get the results of the outputs
            results = pd.DataFrame(
                outputs['final_results'],
                columns=['x1', 'x2', 'f', 'algorithm', 'total_time'],
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


@cli.command('run-test')
def run_test():
    "Testing"
    print('Test!')


if __name__ == "__main__":
    # Be able to run different commands
    cli()
