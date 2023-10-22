# stdlib
import os
import time

# third party
import click
import numpy as np
import pandas as pd
import torch
import wandb
from pygranso.private.getNvar import getNvarTorch
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from scipy.optimize import dual_annealing
from sklearn.datasets import load_digits, load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

# first party
from deeplifting.models import DeepliftingSkipMLP
from deeplifting.utils import (
    get_devices,
    initialize_vector,
    set_seed,
    train_model_to_output,
)


# Build a utility for loading in the iris dataset with option for a test set
def build_iris_dataset(
    num_features=2, species_class=0, test_split=True, torch_version=False
):
    # Load the dataset
    iris = load_iris()

    # The data and target labels
    data = iris.data
    labels = iris.target

    df = pd.DataFrame(data=data, columns=['f1', 'f2', 'f3', 'f4'])
    df['f5'] = 1.0
    df['labels'] = iris.target

    # Resample the data
    df = df.sample(frac=1.0).reset_index(drop=True)

    # Change features here
    if num_features == 2:
        columns = ['f1', 'f2', 'f5']
    elif num_features == 5:
        columns = ['f1', 'f2', 'f3', 'f4', 'f5']

    # Set up the variables
    y = df['labels'].values

    # Binarize the labels
    labels = np.zeros(len(y))
    labels[y != species_class] = -1
    labels[y == species_class] = 1
    y = labels.copy()

    X = df[columns].values

    # Sample the data into train and test
    if test_split:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if torch_version:
            device = torch.device('cpu')
            # Torch X and y
            X_train = torch.from_numpy(X_train).to(device=device, dtype=torch.double)
            X_test = torch.from_numpy(X_test).to(device=device, dtype=torch.double)

            # y variables
            y_train = torch.from_numpy(y_train).to(device=device, dtype=torch.double)
            y_test = torch.from_numpy(y_test).to(device=device, dtype=torch.double)

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
        }

    else:
        X = X.copy()
        y = y.copy()

        if torch_version:
            X = torch.from_numpy(X).to(device=device, dtype=torch.double)
            y = torch.from_numpy(y).to(device=device, dtype=torch.double)

        return {'X_train': X, 'y_train': y, 'X_test': None, 'y_test': None}


def build_mnist_dataset(number_class=0, test_split=True, torch_version=False):
    # Load the dataset
    # Load the MNIST dataset
    digits = load_digits()

    # Split the dataset into features and target variable
    X = digits.data / 255.0

    columns = [f'{i + 1}' for i in range(X.shape[1])]
    df = pd.DataFrame(data=X, columns=columns)
    df[f'f{X.shape[1] + 1}'] = 1.0

    # Set up the variables
    X = df.values
    y = digits.target

    # Binarize the labels
    labels = np.zeros(len(y))
    labels[y != number_class] = -1
    labels[y == number_class] = 1
    y = labels.copy()

    X = df[columns].values

    # Sample the data into train and test
    if test_split:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if torch_version:
            device = torch.device('cpu')
            # Torch X and y
            X_train = torch.from_numpy(X_train).to(device=device, dtype=torch.double)
            X_test = torch.from_numpy(X_test).to(device=device, dtype=torch.double)

            # y variables
            y_train = torch.from_numpy(y_train).to(device=device, dtype=torch.double)
            y_test = torch.from_numpy(y_test).to(device=device, dtype=torch.double)

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
        }

    else:
        X = X.copy()
        y = y.copy()

        if torch_version:
            X = torch.from_numpy(X).to(device=device, dtype=torch.double)
            y = torch.from_numpy(y).to(device=device, dtype=torch.double)

        return {'X_train': X, 'y_train': y, 'X_test': None, 'y_test': None}


def build_cifar100_dataset(test_split=True, torch_version=False):
    # Transformations applied to the dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Load the CIFAR-100 dataset
    cifar100_dataset = datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform
    )

    # Split the dataset into features and target variable
    X = np.array([image.flatten().numpy() for image, label in cifar100_dataset])
    y = np.array([label for _, label in cifar100_dataset])

    columns = [f'{i + 1}' for i in range(X.shape[1])]
    df = pd.DataFrame(data=X, columns=columns)
    df[f'f{X.shape[1] + 1}'] = 1.0
    df['labels'] = y

    # Need a smaller sample
    df = df.sample(frac=0.20)

    X = df[columns + [f'f{X.shape[1] + 1}']].values
    y = df['labels']
    classes = y.isin([71, 72, 73, 74, 75]).values
    y = y.values

    labels = np.zeros(len(y))
    labels[classes] = 1
    labels[~classes] = -1
    y = labels.copy()

    # Sample the data into train and test
    if test_split:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if torch_version:
            device = torch.device('cpu')
            # Convert X and y to Torch tensors
            X_train = torch.tensor(X_train).to(device=device, dtype=torch.double)
            X_test = torch.tensor(X_test).to(device=device, dtype=torch.double)
            y_train = torch.tensor(y_train).to(device=device, dtype=torch.double)
            y_test = torch.tensor(y_test).to(device=device, dtype=torch.double)

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
        }

    else:
        X = X.copy()
        y = y.copy()

        if torch_version:
            device = torch.device('cpu')
            X = torch.tensor(X).to(device=device, dtype=torch.float32)
            y = torch.tensor(y).to(device=device, dtype=torch.long)

        return {'X_train': X, 'y_train': y, 'X_test': None, 'y_test': None}


# Set up the learning function this will be for algorithms such
# as dual annealing
def svm_numpy_objective(weight_vec, inputs_X, labels):
    # Compute SVM objective
    denominator = np.linalg.norm(weight_vec, ord=2)
    prod = np.matmul(weight_vec.T, inputs_X)

    numerator = (labels * prod).flatten()
    obj = numerator / denominator

    # Orig obj
    f = np.amax(-1 * obj)
    return f


# Set up the learning function - this will be for PyGRANSO
def svm_pygranso_objective(X_struct, inputs_X, labels):
    weight_vec = X_struct.w

    # Compute SVM objective
    denominator = torch.linalg.norm(weight_vec, ord=2)
    prod = torch.matmul(weight_vec.T, inputs_X)
    numerator = labels * prod
    obj = numerator / denominator

    # Orig obj
    f = torch.amax(-1 * obj)

    ce = None
    ci = None
    return f, ci, ce


# Set up the learning function
def svm_deeplifting_objective(
    model,
    inputs,
    inputs_X,
    labels,
):
    model.train()
    outputs = model(inputs=inputs)
    weight_vec = outputs.mean(axis=0)

    # Compute SVM objective
    denominator = torch.linalg.norm(weight_vec, ord=2)
    prod = torch.matmul(weight_vec.reshape(1, -1), inputs_X)
    numerator = labels * prod
    obj = numerator / denominator

    # Orig obj
    f = torch.amax(-1 * obj)
    f = f * torch.sqrt(torch.tensor(inputs_X.shape[0]))

    ce = None
    ci = None

    return f, ci, ce


def svm_dual_annealing(X, labels, trial):
    # Set seed for the generated x0
    set_seed(trial)

    # For this problem we will set up arbitrary bounds
    bounds = [(-100, 100)] * X.shape[0]

    # Initialize a weight vector
    x0 = initialize_vector(size=X.shape[0], bounds=None)

    # Setup the objective function
    fn = lambda w: svm_numpy_objective(w, X, labels)

    # Get the result
    result = dual_annealing(
        fn,
        bounds,
        x0=x0,
        maxiter=1,
    )
    return result


def svm_pygranso(X, labels, trial):
    # Set the seed for the random initialization
    set_seed(trial)

    # Get the device
    device = get_devices()

    # Initialize a weight vector
    x0 = initialize_vector(size=X.shape[0], bounds=None)
    x0 = x0.reshape(X.shape[0], 1)
    x0 = torch.from_numpy(x0).to(device=device, dtype=torch.double)

    # Get the dimension of the input variable
    var_in = {"w": list(x0.shape)}

    comb_fn = lambda X_struct: svm_pygranso_objective(
        X_struct,
        X,
        labels,
    )

    opts = pygransoStruct()

    # PyGranso options
    # Increase max number of iterations and let convege to stationarity
    # Do we see local minima in the PyGranso version
    # Dual Annealing, SCIP and Deeplifting, PyGranso (showing there are local minima)
    opts.x0 = torch.reshape(x0, (-1, 1))
    opts.torch_device = device
    opts.print_frequency = 10
    opts.limited_mem_size = 100
    opts.stat_l2_model = False
    opts.double_precision = True
    opts.opt_tol = 1e-5
    opts.maxit = 10000

    # Run the main algorithm
    soln = pygranso(var_spec=var_in, combined_fn=comb_fn, user_opts=opts)
    return soln


def svm_deeplifting(model, data, inputs, trial):
    X_train = data['X_train']
    y_train = data['y_train']

    # Deeplifting time!
    device = get_devices()
    nvar = getNvarTorch(model.parameters())

    # Setup the options
    opts = pygransoStruct()

    # Inital x0
    x0 = (
        torch.nn.utils.parameters_to_vector(model.parameters())
        .detach()
        .reshape(nvar, 1)
        .to(device=device, dtype=torch.double)
    )

    # PyGranso options
    # Increase max number of iterations and
    # let convege to stationarity
    # Do we see local minima in the PyGranso version
    # Dual Annealing, SCIP and Deeplifting,
    # PyGranso (showing there are local minima)
    opts.x0 = x0
    opts.torch_device = device
    opts.print_frequency = 1
    opts.limited_mem_size = 100
    opts.stat_l2_model = False
    opts.double_precision = True
    opts.opt_tol = 1e-5
    opts.maxit = 10000

    # Combined function
    comb_fn = lambda model: svm_deeplifting_objective(
        model,
        inputs,
        X_train.T,
        y_train,
    )  # noqa

    # Run the main algorithm
    model.train()
    soln = pygranso(var_spec=model, combined_fn=comb_fn, user_opts=opts)
    return soln, model


def build_predictions(w, X, version='numpy'):
    """
    Simple utility function to build the predictions
    from the fitted weights
    """
    predictions = w @ X
    if version == 'pytorch':
        predictions = predictions.cpu().detach().numpy()

    predictions = np.sign(predictions)
    return predictions


# Define CLI
@click.group()
def cli():
    pass


@cli.command('run-svm')
@click.option('--algorithm', default='dual-annealing')
@click.option('--trials', default=10)
@click.option('--experimentation', default=True)
def run_svm(algorithm, trials, experimentation):
    """
    Function that will run dual annealing for determining
    the weights for SVM
    """
    if experimentation:
        wandb.login(key='2080070c4753d0384b073105ed75e1f46669e4bf')

        wandb.init(
            # set the wandb project where this run will be logged
            project="Deeplifting-SVM",
            tags=[f'{algorithm}-svm'],
        )

    print('Run Dual Annealing for SVM')

    # Path for the UUID file under the experiments directory
    experiments_path = './experiments'
    uuid_file_path = os.path.join(experiments_path, 'current_experiment_uuid.txt')
    with open(uuid_file_path) as f:
        uuid = f.readline()

    # Load in the CIFAR 100 dataset
    # Numpy data
    if algorithm in ('dual-annealing'):
        data = build_cifar100_dataset(test_split=True, torch_version=False)
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']

    # If the algorithm is either pygranso or deeplifting we need to
    # convert the data
    if algorithm in ('pygranso'):
        # Get the device
        device = get_devices()

        # Load the torch data
        data = build_cifar100_dataset(test_split=True, torch_version=True)
        X_train = data['X_train'].to(device=device, dtype=torch.double)
        y_train = data['y_train'].to(device=device, dtype=torch.double)
        X_test = data['X_test'].to(device=device, dtype=torch.double)
        y_test = data['y_test'].to(device=device, dtype=torch.double)

    # result data
    results_df_list = []

    # We want to run n trials of the modeling for analysis
    for trial in range(trials):
        print(f'Running trial {trial + 1}')

        # Run the dual annealing version
        start = time.time()

        if algorithm == 'dual-annealing':
            result = svm_dual_annealing(X_train.T, y_train, trial=trial)
            objective = result.fun
            weights = result.x

            # Create predictions for train and test data
            preds_train = build_predictions(weights, X_train.T)
            preds_test = build_predictions(weights, X_test.T)

            end = time.time()

            # Compute total time
            total_time = end - start

            # Train and test data accuracy
            train_accuracy = accuracy_score(y_train, preds_train)
            test_accuracy = accuracy_score(y_test, preds_test)

        elif algorithm == 'pygranso':
            result = svm_pygranso(X_train.T, y_train, trial=trial)
            objective = result.best.f
            weights = result.best.x.flatten().reshape(1, -1)

            # Create predictions for train and test data
            preds_train = build_predictions(
                weights, X_train.T, version='pytorch'
            ).flatten()
            preds_test = build_predictions(
                weights, X_test.T, version='pytorch'
            ).flatten()

            end = time.time()

            # Compute total time
            total_time = end - start

            # Train and test data accuracy

            train_accuracy = accuracy_score(
                y_train.cpu().numpy().flatten(), preds_train
            )
            test_accuracy = accuracy_score(y_test.cpu().numpy().flatten(), preds_test)

        # save the data
        results_df = pd.DataFrame(
            {
                'values': [objective, train_accuracy, test_accuracy],
                'metric': ['Objective', 'Train-Accuracy', 'Test-Accuracy'],
            }
        )
        results_df['trial'] = trial
        results_df['problem_name'] = 'CIFAR-100'
        results_df['total_time'] = total_time
        results_df['algorithm'] = algorithm

        # Append data to list
        results_df_list.append(results_df)

    # Create full dataframe
    results_df = pd.concat(results_df_list)

    # Save the data
    svm_path = f'./experiments/{uuid}/svm/{algorithm}'
    file_name = 'svm.parquet'
    save_path = os.path.join(svm_path, file_name)

    # Save dual annealing data
    print('Saving data!')
    results_df.to_parquet(save_path)

    print('Process finished!')
    if experimentation:
        wandb.finish()


@cli.command('run-svm-deeplifting')
@click.option('--trials', default=10)
@click.option('--experimentation', default=True)
def run_svm_deeplifting(trials, experimentation):
    """
    Function that will run deeplifting for determining
    the weights for SVM
    """
    if experimentation:
        wandb.login(key='2080070c4753d0384b073105ed75e1f46669e4bf')

        wandb.init(
            # set the wandb project where this run will be logged
            project="Deeplifting-SVM",
            tags=['deeplifting-pygranso-svm'],
        )

    print('Run Deeplifting-PyGranso for SVM')

    # Path for the UUID file under the experiments directory
    experiments_path = './experiments'
    uuid_file_path = os.path.join(experiments_path, 'current_experiment_uuid.txt')
    with open(uuid_file_path) as f:
        uuid = f.readline()

    # Load in the CIFAR 100 dataset
    # Get the device
    device = get_devices()

    # Load the torch data
    data = build_cifar100_dataset(test_split=True, torch_version=True)
    X_train = data['X_train'].to(device=device, dtype=torch.double)
    y_train = data['y_train'].to(device=device, dtype=torch.double)
    X_test = data['X_test'].to(device=device, dtype=torch.double)
    y_test = data['y_test'].to(device=device, dtype=torch.double)

    # data for deeplifting
    dl_data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
    }

    # result data
    results_df_list = []

    # We want to run n trials of the modeling for analysis
    for trial in range(trials):
        set_seed(trial)
        print(f'Running trial {trial + 1}')

        # Get the inputs for the model
        inputs = torch.randn(1, 5 * X_train.T.shape[0])
        inputs = inputs.to(device=device, dtype=torch.double)

        # Get the initial seed
        x0 = initialize_vector(size=X_train.T.shape[0], bounds=None)
        x0 = x0.reshape(X_train.T.shape[0], 1)
        x0 = torch.from_numpy(x0).to(device=device, dtype=torch.double)

        # Initialize the deeplifting model
        model = DeepliftingSkipMLP(
            input_size=1,
            hidden_sizes=(256,) * 3,
            output_size=X_train.T.shape[0],
            bounds=None,
            skip_every_n=1,
            activation='sine',
            output_activation='sine',
            agg_function='sum',
            include_bn=False,
            seed=trial,
        )

        # Put the model on the correct device
        model = model.to(device=device, dtype=torch.double)

        print('Set weights to match x0')
        train_model_to_output(
            inputs=inputs, model=model, x0=x0, epochs=100000, lr=1e-4, tolerance=1e-10
        )

        # Run the dual annealing version
        start = time.time()

        # put model in training mode
        result, model = svm_deeplifting(
            model=model, data=dl_data, inputs=inputs, trial=trial
        )
        objective = result.best.f

        # Weights are different with deeplifting
        model.eval()
        weights = model(inputs=inputs)
        weights = weights.mean(axis=0).reshape(1, -1)

        # Create predictions for train and test data
        preds_train = build_predictions(weights, X_train.T, version='pytorch').flatten()
        preds_test = build_predictions(weights, X_test.T, version='pytorch').flatten()

        end = time.time()

        # Compute total time
        total_time = end - start

        # Train and test data accuracy
        train_accuracy = accuracy_score(y_train.cpu().numpy().flatten(), preds_train)
        test_accuracy = accuracy_score(y_test.cpu().numpy().flatten(), preds_test)

        # save the data
        results_df = pd.DataFrame(
            {
                'values': [objective, train_accuracy, test_accuracy],
                'metric': ['Objective', 'Train-Accuracy', 'Test-Accuracy'],
            }
        )
        results_df['trial'] = trial
        results_df['problem_name'] = 'CIFAR-100'
        results_df['total_time'] = total_time
        results_df['algorithm'] = 'deeplifting-pygranso'

        # Append data to list
        results_df_list.append(results_df)

    # Create full dataframe
    results_df = pd.concat(results_df_list)

    # Save the data
    svm_path = f'./experiments/{uuid}/svm/deeplifting-pygranso'
    file_name = 'svm.parquet'
    save_path = os.path.join(svm_path, file_name)

    # Save dual annealing data
    print('Saving data!')
    results_df.to_parquet(save_path)

    print('Process finished!')
    if experimentation:
        wandb.finish()


if __name__ == "__main__":
    # Be able to run different commands
    cli()
