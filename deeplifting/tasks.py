# stdlib
import os

# third party
import click
import neptune

# first party
from deeplifting.optimization import run_deeplifting
from deeplifting.problems import PROBLEMS_BY_NAME
from deeplifting.utils import create_optimization_plot


@click.command('run-deeplifting')
@click.option("--problem_name", default="ackley", type=click.STRING)
def run_deeplifting_task(problem_name):
    """
    Function to run the deeplifting task.
    """
    # Get the problem details
    problem = PROBLEMS_BY_NAME[problem_name]

    # Print the problem name to stdout
    click.echo(problem_name)

    # Enable the neptune run
    # Get api token
    # TODO: If api token is not present log a warning
    # and default to saving files locally
    neptune_api_token = os.environ.get('NEPTUNE_API_TOKEN')

    run = neptune.init_run(
        project="dever120/Deeplifting",
        api_token=neptune_api_token,
    )

    # Get the deeplifting outputs
    # TODO: Will make the arguments of the model configurable
    # So we can use hyperopt to test different configurations
    # of the model
    outputs = run_deeplifting(problem, trials=10)

    # Get final results
    results = outputs['final_results']

    # Log the image and results
    fig = create_optimization_plot(problem_name, problem, results, colormap='autumn_r')
    run[f"deeplifting-{problem.name}-final-results-surface-and-contour"].upload(fig)
