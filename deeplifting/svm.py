# third party
import numpy as np
import pandas as pd
import torch
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
from deeplifting.utils import get_devices


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


def build_cifar100_dataset(image_class=46, test_split=True, torch_version=False):
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
    y = df['labels'].values
    labels = np.zeros(len(y))
    labels[y == image_class] = 1
    labels[y != image_class] = -1
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
def numpy_svm(weight_vec, inputs_X, labels):
    # Compute SVM objective
    denominator = np.linalg.norm(weight_vec, ord=2)
    prod = np.matmul(weight_vec.T, inputs_X)

    numerator = (labels * prod).flatten()
    obj = numerator / denominator

    # Orig obj
    f = np.amax(-1 * obj)
    return f


# Set up the learning function - this will be for PyGRANSO
def pygranso_svm(X_struct, inputs_X, labels):
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
def deeplifting_svm(model, X, labels):
    outputs = model(None)
    weight_vec = outputs.mean(axis=0)

    # Compute SVM objective
    denominator = torch.linalg.norm(weight_vec, ord=2)
    prod = torch.matmul(weight_vec.T, X)
    numerator = labels * prod
    obj = numerator / denominator

    # Orig obj
    f = torch.amax(-1 * obj)

    ce = None
    ci = None
    return f, ci, ce


def run_dual_annealing_svm(X, labels):
    # Initialize a weight vector
    x0 = np.random.randn(X.shape[0])

    # Setup the objective function
    fn = lambda w: numpy_svm(w, X, labels)

    # For this problem we will set up arbitrary bounds
    bounds = [(-10, 10)] * X.shape[0]

    # Get the result
    result = dual_annealing(
        fn,
        bounds,
        x0=x0,
        maxiter=1000,
    )
    return result


def run_pygranso(X, labels):
    device = torch.device('cpu')
    w0 = torch.randn(
        (X.shape[0], 1),
    ).to(device, dtype=torch.double)
    var_in = {"w": list(w0.shape)}

    comb_fn = lambda X_struct: pygranso_svm(
        X_struct,
        X,
        labels,
    )

    opts = pygransoStruct()

    # PyGranso options
    # Increase max number of iterations and let convege to stationarity
    # Do we see local minima in the PyGranso version
    # Dual Annealing, SCIP and Deeplifting, PyGranso (showing there are local minima)
    opts.x0 = torch.reshape(w0, (-1, 1))
    opts.torch_device = device
    opts.print_frequency = 10
    opts.limited_mem_size = 5
    opts.stat_l2_model = False
    opts.double_precision = True
    opts.opt_tol = 1e-5
    opts.maxit = 10000

    # Run the main algorithm
    soln = pygranso(var_spec=var_in, combined_fn=comb_fn, user_opts=opts)
    return soln


def run_deeplifting(model, X, labels):
    # Deeplifting time!
    device = get_devices()
    model = model.to(device=device, dtype=torch.double)
    nvar = getNvarTorch(model.parameters())

    opts = pygransoStruct()

    # Inital x0
    x0 = (
        torch.nn.utils.parameters_to_vector(model.parameters())
        .detach()
        .reshape(nvar, 1)
        .to(device=device, dtype=torch.double)
    )

    # PyGranso options
    # Increase max number of iterations and let convege to stationarity
    # Do we see local minima in the PyGranso version
    # Dual Annealing, SCIP and Deeplifting, PyGranso (showing there are local minima)
    opts.x0 = x0
    opts.torch_device = device
    opts.print_frequency = 100
    opts.limited_mem_size = 100
    opts.stat_l2_model = False
    opts.double_precision = True
    opts.opt_tol = 1e-5
    opts.maxit = 6000

    # Combined function
    comb_fn = lambda model: deeplifting_svm(model, X, labels)  # noqa

    # Run the main algorithm
    soln = pygranso(var_spec=model, combined_fn=comb_fn, user_opts=opts)

    return soln


def build_predictions(w, X):
    predictions = np.sign(w @ X)
    return predictions


if __name__ == "__main__":
    # Load in the CIFAR 100 dataset
    # Numpy data
    data = build_cifar100_dataset(image_class=46, test_split=True, torch_version=False)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    # Torch data
    data = build_cifar100_dataset(image_class=46, test_split=True, torch_version=True)
    Xt_train = data['X_train']
    yt_train = data['y_train']
    Xt_test = data['X_test']
    yt_test = data['y_test']

    # Need to put the torch data on the same device
    device = get_devices()
    Xt_train = Xt_train.to(device=device, dtype=torch.double)
    y_train = y_train.to(device=device, dtype=torch.double)
    Xt_test = Xt_test.to(device=device, dtype=torch.double)
    y_test = y_test.to(device=device, dtype=torch.double)

    # Initialize the deeplifting model
    model = DeepliftingSkipMLP(
        input_size=64,
        hidden_sizes=(128,) * 3,
        output_size=Xt_train.T.shape[0],
        bounds=None,
        skip_every_n=1,
        activation='leaky_relu',
        output_activation='sine',
        agg_function='identity',
        include_bn=True,
        seed=0,
    )

    # Run deeplifting and obtain the weights
    dl_result = run_deeplifting(model, Xt_train.T, yt_train)
    dl_weights = model(inputs=None)
    dl_weights = dl_weights.mean(axis=0)
    dl_weights = dl_weights.detach().cpu().numpy().flatten().reshape(1, -1)

    # Train accuracy
    preds_train = build_predictions(dl_weights, X_train.T)

    # Test accuracy
    preds_test = build_predictions(dl_weights, X_test.T)

    # Print metric
    print(
        accuracy_score(y_train, preds_train.flatten()),
        accuracy_score(y_test, preds_test.flatten()),
    )
