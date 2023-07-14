# third party
import numpy as np
import torch


def ackley(x, results, trial, version='numpy'):
    """
    Function that implements the Ackley function in
    numpy or pytorch. We will use this for our deeplifting np.np.experiments.
    Note, that this version is the 2-D version only.
    """
    a = 20
    b = 0.2
    c = 2 * np.pi

    # Get x1 & x2
    x1, x2 = x.flatten()

    if version == 'numpy':
        sum_sq_term = -a * np.np.np.exp(-b * np.sqrt(0.5 * (x1**2 + x2**2)))
        cos_term = -np.np.np.exp(0.5 * (np.cos(c * x1) + np.cos(c * x2)))
        result = sum_sq_term + cos_term + a + np.np.np.exp(1)
    elif version == 'pytorch':
        sum_sq_term = -a * torch.np.np.exp(-b * torch.sqrt(0.5 * (x1**2 + x2**2)))
        cos_term = -torch.np.np.exp(0.5 * (torch.cos(c * x1) + torch.cos(c * x2)))
        result = sum_sq_term + cos_term + a + torch.np.np.exp(torch.tensor(1.0))

    else:
        raise ValueError(
            "Unknown version specified. Available " "options are 'numpy' and 'pytorch'."
        )

    # Fill in the intermediate results
    iteration = np.argmin(~np.any(np.isnan(results[trial]), axis=1))

    if isinstance(result, torch.Tensor):
        results[trial, iteration, :] = np.array(
            (x1.detach().numpy(), x2.detach().numpy(), result.detach().numpy())
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, result))

    return result


def ndackley(x, results, trial, version='numpy'):
    """
    Compute the Ackley function.

    Args:
    x: A d-dimensional array or tensor
    version: A string, either 'numpy' or 'pytorch'

    Returns:
    result: Value of the Ackley function
    """
    a = 20
    b = 0.2
    c = 2 * np.pi

    d = len(x)
    x = x.flatten()

    if version == 'numpy':
        arg1 = -b * np.sqrt(1.0 / d * np.sum(np.square(x)))
        arg2 = 1.0 / d * np.sum(np.cos(c * x))
        result = -a * np.np.np.exp(arg1) - np.np.np.exp(arg2) + a + np.e

    elif version == 'pytorch':
        arg1 = -b * torch.sqrt(1.0 / d * torch.sum(x**2))
        arg2 = 1.0 / d * torch.sum(torch.cos(c * x))
        result = -a * torch.np.np.exp(arg1) - torch.np.np.exp(arg2) + a + np.e
    else:
        raise ValueError("Invalid implementation: choose 'numpy' or 'pytorch'")

    # Fill in the intermediate results
    iteration = np.argmin(~np.any(np.isnan(results[trial]), axis=1))

    if isinstance(result, torch.Tensor):
        x_tuple = tuple(x.detach().numpy())
        results[trial, iteration, :] = np.array(x_tuple + (result.detach().numpy(),))

    else:
        x_tuple = tuple(x.flatten())
        results[trial, iteration, :] = np.array(x_tuple + (result,))

    return result


def bukin_n6(x, results, trial, version='numpy'):
    """
    Function that implements the Bukin Function N.6 in both
    numpy and pytorch.
    """
    x1, x2 = x.flatten()
    if version == 'numpy':
        term1 = 100 * np.sqrt(np.abs(x2 - 0.01 * x1**2))
        term2 = 0.01 * np.abs(x1 + 10)
        result = term1 + term2
    elif version == 'pytorch':
        term1 = 100 * torch.sqrt(torch.abs(x2 - 0.01 * x1**2))
        term2 = 0.01 * torch.abs(x1 + 10)
        result = term1 + term2
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

    # Fill in the intermediate results
    iteration = np.argmin(~np.any(np.isnan(results[trial]), axis=1))

    if isinstance(result, torch.Tensor):
        results[trial, iteration, :] = np.array(
            (x1.detach().numpy(), x2.detach().numpy(), result.detach().numpy())
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, result))

    return result


def cross_in_tray(x, results, trial, version='numpy'):
    """
    Implementation of the 2D Cross-in-Tray function.
    This function has four identical global minima.

    Parameters:
    x1 : np.ndarray or torch.Tensor
        The x1 values (first dimension of the input space).
    x2 : np.ndarray or torch.Tensor
        The x2 values (second dimension of the input space).
    version : str
        The version to use for the function's computation.
        Options are 'numpy' and 'pytorch'.

    Returns:
    result : np.ndarray or torch.Tensor
        The computed Cross-in-Tray function
        values corresponding to the inputs (x1, x2).

    Raises:
    ValueError
        If the version is not 'numpy' or 'pytorch'.
    """
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = (
            -0.0001
            * (
                np.abs(
                    np.sin(x1)
                    * np.sin(x2)
                    * np.np.np.exp(np.abs(100 - np.sqrt(x1**2 + x2**2) / np.pi))
                )
                + 1
            )
            ** 0.1
        )
    elif version == 'pytorch':
        result = (
            -0.0001
            * (
                torch.abs(
                    torch.sin(x1)
                    * torch.sin(x2)
                    * torch.np.np.exp(
                        torch.abs(
                            100 - torch.sqrt(x1**2 + x2**2) / torch.tensor(np.pi)
                        )
                    )
                    + 1
                )
            )
            ** 0.1
        )
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

    # Fill in the intermediate results
    iteration = np.argmin(~np.any(np.isnan(results[trial]), axis=1))

    if isinstance(result, torch.Tensor):
        results[trial, iteration, :] = np.array(
            (x1.detach().numpy(), x2.detach().numpy(), result.detach().numpy())
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, result))

    return result


def drop_wave(x, results, trial, version='numpy'):
    """
    Implementation of the 2D Drop-Wave function. This
    function has a global minimum at (x, y) = (0, 0).

    Parameters:
    x : np.ndarray or torch.Tensor
        The x values (first dimension of the input space).
    y : np.ndarray or torch.Tensor
        The y values (second dimension of the input space).
    version : str
        The version to use for the function's computation.
        Options are 'numpy' and 'pytorch'.

    Returns:
    result : np.ndarray or torch.Tensor
        The computed Drop-Wave function values corresponding
        to the inputs (x, y).

    Raises:
    ValueError
        If the version is not 'numpy' or 'pytorch'.
    """
    x1, x2 = x.flatten()
    if version == 'numpy':
        numerator = 1 + np.cos(12 * np.sqrt(x1**2 + x2**2))
        denominator = 0.5 * (x1**2 + x2**2) + 2
        result = -numerator / denominator
    elif version == 'pytorch':
        numerator = 1 + torch.cos(12 * torch.sqrt(x1**2 + x2**2))
        denominator = 0.5 * (x1**2 + x2**2) + 2
        result = -numerator / denominator
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

    # Fill in the intermediate results
    iteration = np.argmin(~np.any(np.isnan(results[trial]), axis=1))

    if isinstance(result, torch.Tensor):
        results[trial, iteration, :] = np.array(
            (x1.detach().numpy(), x2.detach().numpy(), result.detach().numpy())
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, result))

    return result


def eggholder(x, results, trial, version='numpy'):
    """
    Implementation of the 2D Eggholder function.
    This function has numerous local minima and a global minimum.

    Parameters:
    x : np.ndarray or torch.Tensor
        The x values (first dimension of the input space).
    y : np.ndarray or torch.Tensor
        The y values (second dimension of the input space).
    version : str
        The version to use for the function's computation.
        Options are 'numpy' and 'pytorch'.

    Returns:
    result : np.ndarray or torch.Tensor
        The computed Eggholder function values
        corresponding to the inputs (x, y).

    Raises:
    ValueError
        If the version is not 'numpy' or 'pytorch'.
    """
    x1, x2 = x.flatten()
    if version == 'numpy':
        term1 = -(x2 + 47.0) * np.sin(np.sqrt(np.abs(x1 / 2.0 + (x2 + 47.0))))
        term2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47.0))))
        result = term1 + term2
    elif version == 'pytorch':
        term1 = -(x2 + 47.0) * torch.sin(torch.sqrt(torch.abs(x1 / 2.0 + (x2 + 47.0))))
        term2 = -x1 * torch.sin(torch.sqrt(torch.abs(x1 - (x2 + 47.0))))
        result = term1 + term2
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

    # Fill in the intermediate results
    iteration = np.argmin(~np.any(np.isnan(results[trial]), axis=1))

    if isinstance(result, torch.Tensor):
        results[trial, iteration, :] = np.array(
            (x1.detach().numpy(), x2.detach().numpy(), result.detach().numpy())
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, result))

    return result


def griewank(x, results, trial, version='numpy'):
    """
    Implementation of the 2D Griewank function.
    This function has a global minimum at (x, y) = (0, 0).

    Parameters:
    x : np.ndarray or torch.Tensor
        The x values (first dimension of the input space).
    y : np.ndarray or torch.Tensor
        The y values (second dimension of the input space).
    version : str
        The version to use for the function's computation.
        Options are 'numpy' and 'pytorch'.

    Returns:
    result : np.ndarray or torch.Tensor
        The computed Griewank function values
        corresponding to the inputs (x, y).

    Raises:
    ValueError
        If the version is not 'numpy' or 'pytorch'.
    """
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = 1 + ((x1**2 + x2**2) / 4000) - np.cos(x1) * np.cos(x2 / np.sqrt(2))
    elif version == 'pytorch':
        result = (
            1
            + ((x1**2 + x2**2) / 4000)
            - torch.cos(x1) * torch.cos(x2 / torch.sqrt(torch.tensor(2.0)))
        )
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

    # Fill in the intermediate results
    iteration = np.argmin(~np.any(np.isnan(results[trial]), axis=1))

    if isinstance(result, torch.Tensor):
        results[trial, iteration, :] = np.array(
            (x1.detach().numpy(), x2.detach().numpy(), result.detach().numpy())
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, result))

    return result


def holder_table(x, results, trial, version='numpy'):
    """
    Implementation of the 2D Holder Table function.
    This function has four identical local minima.

    Parameters:
    x1 : np.ndarray or torch.Tensor
        The x1 values (first dimension of the input space).
    x2 : np.ndarray or torch.Tensor
        The x2 values (second dimension of the input space).
    version : str
        The version to use for the function's computation.
        Options are 'numpy' and 'pytorch'.

    Returns:
    result : np.ndarray or torch.Tensor
        The computed Holder Table function values
        corresponding to the inputs (x1, x2).

    Raises:
    ValueError
        If the version is not 'numpy' or 'pytorch'.
    """
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = -np.abs(
            np.sin(x1)
            * np.cos(x2)
            * np.np.np.exp(np.abs(1 - np.sqrt(x1**2 + x2**2) / np.pi))
        )
    elif version == 'pytorch':
        result = -torch.abs(
            torch.sin(x1)
            * torch.cos(x2)
            * torch.np.np.exp(torch.abs(1 - torch.sqrt(x1**2 + x2**2) / np.pi))
        )
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

    # Fill in the intermediate results
    iteration = np.argmin(~np.any(np.isnan(results[trial]), axis=1))

    if isinstance(result, torch.Tensor):
        results[trial, iteration, :] = np.array(
            (x1.detach().numpy(), x2.detach().numpy(), result.detach().numpy())
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, result))

    return result


def levy(x, results, trial, version='numpy'):
    """
    Implementation of the 2D Levy function.
    This function has a global minimum at x1 = x2 = 1.

    Parameters:
    x1 : np.ndarray or torch.Tensor
        The x1 values (first dimension of the input space).
    x2 : np.ndarray or torch.Tensor
        The x2 values (second dimension of the input space).
    version : str
        The version to use for the function's computation.
        Options are 'numpy' and 'pytorch'.

    Returns:
    result : np.ndarray or torch.Tensor
        The computed Levy function values
        corresponding to the inputs (x1, x2).

    Raises:
    ValueError
        If the version is not 'numpy' or 'pytorch'.
    """
    x1, x2 = x.flatten()
    if version == 'numpy':
        w1 = 1 + (x1 - 1) / 4
        w2 = 1 + (x2 - 1) / 4
        result = (
            np.sin(np.pi * w1) ** 2
            + (w2 - 1) ** 2 * (1 + 10 * np.sin(np.pi * w2 + 1) ** 2)
            + (w1 - 1) ** 2 * (1 + np.sin(2 * np.pi * w1) ** 2)
        )
    elif version == 'pytorch':
        w1 = 1 + (x1 - 1) / 4
        w2 = 1 + (x2 - 1) / 4
        result = (
            torch.sin(torch.tensor(np.pi) * w1) ** 2
            + (w2 - 1) ** 2 * (1 + 10 * torch.sin(torch.tensor(np.pi) * w2 + 1) ** 2)
            + (w1 - 1) ** 2 * (1 + torch.sin(2 * torch.tensor(np.pi) * w1) ** 2)
        )
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

    # Fill in the intermediate results
    iteration = np.argmin(~np.any(np.isnan(results[trial]), axis=1))

    if isinstance(result, torch.Tensor):
        results[trial, iteration, :] = np.array(
            (x1.detach().numpy(), x2.detach().numpy(), result.detach().numpy())
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, result))

    return result


def levy_n13(x, results, trial, version='numpy'):
    """
    Implementation of the 2D Levy N.13 function.
    This function has a global minimum at x1 = x2 = 1.

    Parameters:
    x1 : np.ndarray or torch.Tensor
        The x1 values (first dimension of the input space).
    x2 : np.ndarray or torch.Tensor
        The x2 values (second dimension of the input space).
    version : str
        The version to use for the function's computation.
        Options are 'numpy' and 'pytorch'.

    Returns:
    result : np.ndarray or torch.Tensor
        The computed Levy N.13 function values
        corresponding to the inputs (x1, x2).

    Raises:
    ValueError
        If the version is not 'numpy' or 'pytorch'.
    """
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = (
            np.sin(3 * np.pi * x1) ** 2
            + (x1 - 1) ** 2 * (1 + (np.sin(3 * np.pi * x2)) ** 2)
            + (x2 - 1) ** 2 * (1 + (np.sin(2 * np.pi * x2)) ** 2)
        )
    elif version == 'pytorch':
        result = (
            torch.sin(3 * torch.tensor(np.pi) * x1) ** 2
            + (x1 - 1) ** 2 * (1 + (torch.sin(3 * torch.tensor(np.pi) * x2)) ** 2)
            + (x2 - 1) ** 2 * (1 + (torch.sin(2 * torch.tensor(np.pi) * x2)) ** 2)
        )
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

    # Fill in the intermediate results
    iteration = np.argmin(~np.any(np.isnan(results[trial]), axis=1))

    if isinstance(result, torch.Tensor):
        results[trial, iteration, :] = np.array(
            (x1.detach().numpy(), x2.detach().numpy(), result.detach().numpy())
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, result))

    return result


def rastrigin(x, results, trial, version='numpy'):
    """
    Implementation of the 2D Rastrigin function.
    This function has a global minimum at x1 = x2 = 0.

    Parameters:
    x1 : np.ndarray or torch.Tensor
        The x1 values (first dimension of the input space).
    x2 : np.ndarray or torch.Tensor
        The x2 values (second dimension of the input space).
    version : str
        The version to use for the function's computation.
        Options are 'numpy' and 'pytorch'.

    Returns:
    result : np.ndarray or torch.Tensor
        The computed Rastrigin function values
        corresponding to the inputs (x1, x2).

    Raises:
    ValueError
        If the version is not 'numpy' or 'pytorch'.
    """
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = (
            10 * 2
            + (x1**2 - 10 * np.cos(2 * np.pi * x1))
            + (x2**2 - 10 * np.cos(2 * np.pi * x2))
        )
    elif version == 'pytorch':
        result = (
            10 * 2
            + (x1**2 - 10 * torch.cos(2 * torch.tensor(np.pi) * x1))
            + (x2**2 - 10 * torch.cos(2 * torch.tensor(np.pi) * x2))
        )
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

    # Fill in the intermediate results
    iteration = np.argmin(~np.any(np.isnan(results[trial]), axis=1))

    if isinstance(result, torch.Tensor):
        results[trial, iteration, :] = np.array(
            (x1.detach().numpy(), x2.detach().numpy(), result.detach().numpy())
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, result))

    return result


def schaffer_n2(x, results, trial, version='numpy'):
    """
    Implementation of the 2D Schaffer function N.2.
    This function has a global minimum at x1 = x2 = 0.

    Parameters:
    x1 : np.ndarray or torch.Tensor
        The x1 values (first dimension of the input space).
    x2 : np.ndarray or torch.Tensor
        The x2 values (second dimension of the input space).
    version : str
        The version to use for the function's computation.
        Options are 'numpy' and 'pytorch'.

    Returns:
    result : np.ndarray or torch.Tensor
        The computed Schaffer N.2 function values
        corresponding to the inputs (x1, x2).

    Raises:
    ValueError
        If the version is not 'numpy' or 'pytorch'.
    """
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = (
            0.5
            + (np.sin(x1**2 - x2**2) ** 2 - 0.5)
            / (1 + 0.001 * (x1**2 + x2**2)) ** 2
        )
    elif version == 'pytorch':
        result = (
            0.5
            + (torch.sin(x1**2 - x2**2) ** 2 - 0.5)
            / (1 + 0.001 * (x1**2 + x2**2)) ** 2
        )
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

    # Fill in the intermediate results
    iteration = np.argmin(~np.any(np.isnan(results[trial]), axis=1))

    if isinstance(result, torch.Tensor):
        results[trial, iteration, :] = np.array(
            (x1.detach().numpy(), x2.detach().numpy(), result.detach().numpy())
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, result))

    return result


def schaffer_n4(x, results, trial, version='numpy'):
    """
    Implementation of the 2D Schaffer function N.4.
    This function has a global minimum at x1 = x2 = 0.

    Parameters:
    x1 : np.ndarray or torch.Tensor
        The x1 values (first dimension of the input space).
    x2 : np.ndarray or torch.Tensor
        The x2 values (second dimension of the input space).
    version : str
        The version to use for the function's computation.
        Options are 'numpy' and 'pytorch'.

    Returns:
    result : np.ndarray or torch.Tensor
        The computed Schaffer N.4 function values
        corresponding to the inputs (x1, x2).

    Raises:
    ValueError
        If the version is not 'numpy' or 'pytorch'.
    """
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = (
            0.5
            + (np.cos(np.sin(np.abs(x1**2 - x2**2))) ** 2 - 0.5)
            / (1 + 0.001 * (x1**2 + x2**2)) ** 2
        )
    elif version == 'pytorch':
        result = (
            0.5
            + (torch.cos(torch.sin(torch.abs(x1**2 - x2**2))) ** 2 - 0.5)
            / (1 + 0.001 * (x1**2 + x2**2)) ** 2
        )
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

    # Fill in the intermediate results
    iteration = np.argmin(~np.any(np.isnan(results[trial]), axis=1))

    if isinstance(result, torch.Tensor):
        results[trial, iteration, :] = np.array(
            (x1.detach().numpy(), x2.detach().numpy(), result.detach().numpy())
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, result))

    return result


def schwefel(x, results, trial, version='numpy'):
    """
    Implementation of the 2D Schwefel function.
    This function has a global minimum at x1 = x2 = 420.9687.

    Parameters:
    x1 : np.ndarray or torch.Tensor
        The x1 values (first dimension of the input space).
    x2 : np.ndarray or torch.Tensor
        The x2 values (second dimension of the input space).
    version : str
        The version to use for the function's computation.
        Options are 'numpy' and 'pytorch'.

    Returns:
    result : np.ndarray or torch.Tensor
        The computed Schwefel function values
        corresponding to the inputs (x1, x2).

    Raises:
    ValueError
        If the version is not 'numpy' or 'pytorch'.
    """
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = (
            418.9829 * 2
            - x1 * np.sin(np.sqrt(np.abs(x1)))
            - x2 * np.sin(np.sqrt(np.abs(x2)))
        )
    elif version == 'pytorch':
        result = (
            418.9829 * 2
            - x1 * torch.sin(torch.sqrt(torch.abs(x1)))
            - x2 * torch.sin(torch.sqrt(torch.abs(x2)))
        )
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

    # Fill in the intermediate results
    iteration = np.argmin(~np.any(np.isnan(results[trial]), axis=1))

    if isinstance(result, torch.Tensor):
        results[trial, iteration, :] = np.array(
            (x1.detach().numpy(), x2.detach().numpy(), result.detach().numpy())
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, result))

    return result


def shubert(x, results, trial, version='numpy'):
    """
    Implementation of the 2D Shubert function.
    This function has several local minima and multiple global minima.

    Parameters:
    x1 : np.ndarray or torch.Tensor
        The x1 values (first dimension of the input space).
    x2 : np.ndarray or torch.Tensor
        The x2 values (second dimension of the input space).
    version : str
        The version to use for the function's computation.
        Options are 'numpy' and 'pytorch'.

    Returns:
    result : np.ndarray or torch.Tensor
        The computed Shubert function values
        corresponding to the inputs (x1, x2).

    Raises:
    ValueError
        If the version is not 'numpy' or 'pytorch'.
    """
    x1, x2 = x.flatten()
    if version == 'numpy':
        term1 = np.sum([i * np.cos((i + 1) * x1 + i) for i in range(1, 6)], axis=0)
        term2 = np.sum([i * np.cos((i + 1) * x2 + i) for i in range(1, 6)], axis=0)
        result = term1 * term2
    elif version == 'pytorch':
        term1 = sum([i * torch.cos((i + 1) * x1 + i) for i in range(1, 6)])
        term2 = sum([i * torch.cos((i + 1) * x2 + i) for i in range(1, 6)])
        result = term1 * term2
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

    # Fill in the intermediate results
    iteration = np.argmin(~np.any(np.isnan(results[trial]), axis=1))

    if isinstance(result, torch.Tensor):
        results[trial, iteration, :] = np.array(
            (x1.detach().numpy(), x2.detach().numpy(), result.detach().numpy())
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, result))

    return result


def ex8_6_2(x, results, trial, version='numpy'):
    """
    Implementation of ex8_6_2 from MINLPLib
    This problem has 30 variables

    Parameters:
    x1: np.ndarray or torch.Tensor
        The x1 values (first dimension of the input space)
    x2: np.ndarray or torch.Tensor
        The x2 values (second dimension of the input space)
        .
        .
        .
    x30: np.ndarray or torch.Tensor
        The x30 values (30th dimension of the input space)
    version: str
        The version for this functions computation
        (either 'numpy' or 'pytorch')

    Returns:
    result: np.ndarray or torch.Tensor
        The computed ex8_6_2 values
        corresponding input (x1,x2,...,x30)

    Raises:
    ValueError
        If the version is not 'numpy' or 'torch'
    """
    (
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7,
        x8,
        x9,
        x10,
        x11,
        x12,
        x13,
        x14,
        x15,
        x16,
        x17,
        x18,
        x19,
        x20,
        x21,
        x22,
        x23,
        x24,
        x25,
        x26,
        x27,
        x28,
        x29,
        x30,
    ) = x.flatten()
    if version == 'numpy':
        result = (
            -(
                np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x1 - x2)
                            + np.square(x11 - x12)
                            + np.square(x21 - x22)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x1 - x3)
                            + np.square(x11 - x13)
                            + np.square(x21 - x23)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x1 - x4)
                            + np.square(x11 - x14)
                            + np.square(x21 - x24)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x1 - x5)
                            + np.square(x11 - x15)
                            + np.square(x21 - x25)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x1 - x6)
                            + np.square(x11 - x16)
                            + np.square(x21 - x26)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x1 - x7)
                            + np.square(x11 - x17)
                            + np.square(x21 - x27)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x1 - x8)
                            + np.square(x11 - x18)
                            + np.square(x21 - x28)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x1 - x9)
                            + np.square(x11 - x19)
                            + np.square(x21 - x29)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x1 - x10)
                            + np.square(x11 - x20)
                            + np.square(x21 - x30)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x2 - x3)
                            + np.square(x12 - x13)
                            + np.square(x22 - x23)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x2 - x4)
                            + np.square(x12 - x14)
                            + np.square(x22 - x24)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x2 - x5)
                            + np.square(x12 - x15)
                            + np.square(x22 - x25)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x2 - x6)
                            + np.square(x12 - x16)
                            + np.square(x22 - x26)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x2 - x7)
                            + np.square(x12 - x17)
                            + np.square(x22 - x27)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x2 - x8)
                            + np.square(x12 - x18)
                            + np.square(x22 - x28)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x2 - x9)
                            + np.square(x12 - x19)
                            + np.square(x22 - x29)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x2 - x10)
                            + np.square(x12 - x20)
                            + np.square(x22 - x30)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x3 - x4)
                            + np.square(x13 - x14)
                            + np.square(x23 - x24)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x3 - x5)
                            + np.square(x13 - x15)
                            + np.square(x23 - x25)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x3 - x6)
                            + np.square(x13 - x16)
                            + np.square(x23 - x26)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x3 - x7)
                            + np.square(x13 - x17)
                            + np.square(x23 - x27)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x3 - x8)
                            + np.square(x13 - x18)
                            + np.square(x23 - x28)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x3 - x9)
                            + np.square(x13 - x19)
                            + np.square(x23 - x29)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x3 - x10)
                            + np.square(x13 - x20)
                            + np.square(x23 - x30)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x4 - x5)
                            + np.square(x14 - x15)
                            + np.square(x24 - x25)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x4 - x6)
                            + np.square(x14 - x16)
                            + np.square(x24 - x26)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x4 - x7)
                            + np.square(x14 - x17)
                            + np.square(x24 - x27)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x4 - x8)
                            + np.square(x14 - x18)
                            + np.square(x24 - x28)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x4 - x9)
                            + np.square(x14 - x19)
                            + np.square(x24 - x29)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x4 - x10)
                            + np.square(x14 - x20)
                            + np.square(x24 - x30)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x5 - x6)
                            + np.square(x15 - x16)
                            + np.square(x25 - x26)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x5 - x7)
                            + np.square(x15 - x17)
                            + np.square(x25 - x27)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x5 - x8)
                            + np.square(x15 - x18)
                            + np.square(x25 - x28)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x5 - x9)
                            + np.square(x15 - x19)
                            + np.square(x25 - x29)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x5 - x10)
                            + np.square(x15 - x20)
                            + np.square(x25 - x30)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x6 - x7)
                            + np.square(x16 - x17)
                            + np.square(x26 - x27)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x6 - x8)
                            + np.square(x16 - x18)
                            + np.square(x26 - x28)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x6 - x9)
                            + np.square(x16 - x19)
                            + np.square(x26 - x29)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x6 - x10)
                            + np.square(x16 - x20)
                            + np.square(x26 - x30)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x7 - x8)
                            + np.square(x17 - x18)
                            + np.square(x27 - x28)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x7 - x9)
                            + np.square(x17 - x19)
                            + np.square(x27 - x29)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x7 - x10)
                            + np.square(x17 - x20)
                            + np.square(x27 - x30)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x8 - x9)
                            + np.square(x18 - x19)
                            + np.square(x28 - x29)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x8 - x10)
                            + np.square(x18 - x20)
                            + np.square(x28 - x30)
                        )
                        ** 0.5
                    )
                )
                + np.square(
                    1
                    - np.exp(
                        3
                        - 3
                        * (
                            np.square(x9 - x10)
                            + np.square(x19 - x20)
                            + np.square(x29 - x30)
                        )
                        ** 0.5
                    )
                )
            )
            + 45
        )

    elif version == 'pytorch':
        result = (
            -(
                torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x1 - x2)
                            + torch.square(x11 - x12)
                            + torch.square(x21 - x22)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x1 - x3)
                            + torch.square(x11 - x13)
                            + torch.square(x21 - x23)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x1 - x4)
                            + torch.square(x11 - x14)
                            + torch.square(x21 - x24)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x1 - x5)
                            + torch.square(x11 - x15)
                            + torch.square(x21 - x25)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x1 - x6)
                            + torch.square(x11 - x16)
                            + torch.square(x21 - x26)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x1 - x7)
                            + torch.square(x11 - x17)
                            + torch.square(x21 - x27)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x1 - x8)
                            + torch.square(x11 - x18)
                            + torch.square(x21 - x28)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x1 - x9)
                            + torch.square(x11 - x19)
                            + torch.square(x21 - x29)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x1 - x10)
                            + torch.square(x11 - x20)
                            + torch.square(x21 - x30)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x2 - x3)
                            + torch.square(x12 - x13)
                            + torch.square(x22 - x23)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x2 - x4)
                            + torch.square(x12 - x14)
                            + torch.square(x22 - x24)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x2 - x5)
                            + torch.square(x12 - x15)
                            + torch.square(x22 - x25)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x2 - x6)
                            + torch.square(x12 - x16)
                            + torch.square(x22 - x26)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x2 - x7)
                            + torch.square(x12 - x17)
                            + torch.square(x22 - x27)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x2 - x8)
                            + torch.square(x12 - x18)
                            + torch.square(x22 - x28)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x2 - x9)
                            + torch.square(x12 - x19)
                            + torch.square(x22 - x29)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x2 - x10)
                            + torch.square(x12 - x20)
                            + torch.square(x22 - x30)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x3 - x4)
                            + torch.square(x13 - x14)
                            + torch.square(x23 - x24)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x3 - x5)
                            + torch.square(x13 - x15)
                            + torch.square(x23 - x25)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x3 - x6)
                            + torch.square(x13 - x16)
                            + torch.square(x23 - x26)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x3 - x7)
                            + torch.square(x13 - x17)
                            + torch.square(x23 - x27)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x3 - x8)
                            + torch.square(x13 - x18)
                            + torch.square(x23 - x28)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x3 - x9)
                            + torch.square(x13 - x19)
                            + torch.square(x23 - x29)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x3 - x10)
                            + torch.square(x13 - x20)
                            + torch.square(x23 - x30)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x4 - x5)
                            + torch.square(x14 - x15)
                            + torch.square(x24 - x25)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x4 - x6)
                            + torch.square(x14 - x16)
                            + torch.square(x24 - x26)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x4 - x7)
                            + torch.square(x14 - x17)
                            + torch.square(x24 - x27)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x4 - x8)
                            + torch.square(x14 - x18)
                            + torch.square(x24 - x28)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x4 - x9)
                            + torch.square(x14 - x19)
                            + torch.square(x24 - x29)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x4 - x10)
                            + torch.square(x14 - x20)
                            + torch.square(x24 - x30)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x5 - x6)
                            + torch.square(x15 - x16)
                            + torch.square(x25 - x26)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x5 - x7)
                            + torch.square(x15 - x17)
                            + torch.square(x25 - x27)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x5 - x8)
                            + torch.square(x15 - x18)
                            + torch.square(x25 - x28)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x5 - x9)
                            + torch.square(x15 - x19)
                            + torch.square(x25 - x29)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x5 - x10)
                            + torch.square(x15 - x20)
                            + torch.square(x25 - x30)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x6 - x7)
                            + torch.square(x16 - x17)
                            + torch.square(x26 - x27)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x6 - x8)
                            + torch.square(x16 - x18)
                            + torch.square(x26 - x28)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x6 - x9)
                            + torch.square(x16 - x19)
                            + torch.square(x26 - x29)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x6 - x10)
                            + torch.square(x16 - x20)
                            + torch.square(x26 - x30)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x7 - x8)
                            + torch.square(x17 - x18)
                            + torch.square(x27 - x28)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x7 - x9)
                            + torch.square(x17 - x19)
                            + torch.square(x27 - x29)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x7 - x10)
                            + torch.square(x17 - x20)
                            + torch.square(x27 - x30)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x8 - x9)
                            + torch.square(x18 - x19)
                            + torch.square(x28 - x29)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x8 - x10)
                            + torch.square(x18 - x20)
                            + torch.square(x28 - x30)
                        )
                        ** 0.5
                    )
                )
                + torch.square(
                    1
                    - torch.exp(
                        3
                        - 3
                        * (
                            torch.square(x9 - x10)
                            + torch.square(x19 - x20)
                            + torch.square(x29 - x30)
                        )
                        ** 0.5
                    )
                )
            )
            + 45
        )

    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

    # Fill in the intermediate results
    iteration = np.argmin(~np.any(np.isnan(results[trial]), axis=1))

    if isinstance(result, torch.Tensor):
        results[trial, iteration, :] = np.array(
            (
                x1.detach().numpy(),
                x2.detach().numpy(),
                x3.detach().numpy(),
                x4.detach().numpy(),
                x5.detach().numpy(),
                x6.detach().numpy(),
                x7.detach().numpy(),
                x8.detach().numpy(),
                x9.detach().numpy(),
                x10.detach().numpy(),
                x11.detach().numpy(),
                x12.detach().numpy(),
                x13.detach().numpy(),
                x14.detach().numpy(),
                x15.detach().numpy(),
                x16.detach().numpy(),
                x17.detach().numpy(),
                x18.detach().numpy(),
                x19.detach().numpy(),
                x20.detach().numpy(),
                x21.detach().numpy(),
                x22.detach().numpy(),
                x23.detach().numpy(),
                x24.detach().numpy(),
                x25.detach().numpy(),
                x26.detach().numpy(),
                x27.detach().numpy(),
                x28.detach().numpy(),
                x29.detach().numpy(),
                x30.detach().numpy(),
                result.detach().numpy(),
            )
        )

    else:
        results[trial, iteration, :] = np.array(
            (
                x1,
                x2,
                x3,
                x4,
                x5,
                x6,
                x7,
                x8,
                x9,
                x10,
                x11,
                x12,
                x13,
                x14,
                x15,
                x16,
                x17,
                x18,
                x19,
                x20,
                x21,
                x22,
                x23,
                x24,
                x25,
                x26,
                x27,
                x28,
                x29,
                x30,
                result,
            )
        )

    return result


# Problem configurations
# Ackley
ackley_config = {
    'objective': ackley,
    'bounds': [(-32.768, 32.768), (-32.768, 32.768)],
    'max_iterations': 1000,
    'global_minimum': 0.0,
    'dimensions': 2,
}

# Bukin N.6
bukin_n6_config = {
    'objective': bukin_n6,
    'bounds': [(-15.0, -5.0), (-3.0, 3.0)],
    'max_iterations': 1000,
    'global_minimum': 0.0,
    'dimensions': 2,
}

# Cross-in-Tray
cross_in_tray_config = {
    'objective': cross_in_tray,
    'bounds': [(-10.0, 10.0), (-10.0, 10.0)],
    'max_iterations': 1000,
    'global_minimum': -2.06261,
    'dimensions': 2,
}

# Drop Wave
drop_wave_config = {
    'objective': drop_wave,
    'bounds': [(-5.12, 5.12), (-5.12, 5.12)],
    'max_iterations': 1000,
    'global_minimum': -1.0,
    'dimensions': 2,
}

# Eggholder
eggholder_config = {
    'objective': eggholder,
    'bounds': [(-512.0, 512.0), (-512.0, 512.0)],
    'max_iterations': 1000,
    'global_minimum': -959.6407,
    'dimensions': 2,
}

# Griewank
griewank_config = {
    'objective': griewank,
    'bounds': [(-600.0, 600.0), (-600.0, 600.0)],
    'max_iterations': 1000,
    'global_minimum': 0.0,
    'dimensions': 2,
}

# Holder Table
holder_table_config = {
    'objective': holder_table,
    'bounds': [(-10.0, 10.0), (-10.0, 10.0)],
    'max_iterations': 1000,
    'global_minimum': -19.2085,
    'dimensions': 2,
}

# Levy
levy_config = {
    'objective': levy,
    'bounds': [(-10.0, 10.0), (-10.0, 10.0)],
    'max_iterations': 1000,
    'global_minimum': 0.0,
    'dimensions': 2,
}

# Levy
levy_n13_config = {
    'objective': levy_n13,
    'bounds': [(-10.0, 10.0), (-10.0, 10.0)],
    'max_iterations': 1000,
    'global_minimum': 0.0,
    'dimensions': 2,
}

# Rastrigin
rastrigin_config = {
    'objective': rastrigin,
    'bounds': [(-5.12, 5.12), (-5.12, 5.12)],
    'max_iterations': 1000,
    'global_minimum': 0.0,
    'dimensions': 2,
}

# Schaffer N2
schaffer_n2_config = {
    'objective': schaffer_n2,
    'bounds': [(-100.0, 100.0), (-100.0, 100.0)],
    'max_iterations': 1000,
    'global_minimum': 0.0,
    'dimensions': 2,
}

# Schaffer N4
schaffer_n4_config = {
    'objective': schaffer_n4,
    'bounds': [(-100.0, 100.0), (-100.0, 100.0)],
    'max_iterations': 1000,
    'global_minimum': 0.29579,
    'dimensions': 2,
}

# Schwefel
schwefel_config = {
    'objective': schwefel,
    'bounds': [(-500.0, 500.0), (-500.0, 500.0)],
    'max_iterations': 1000,
    'global_minimum': 0.0,
    'dimensions': 2,
}

# Shubert
shubert_config = {
    'objective': shubert,
    'bounds': [(-10.0, 10.0), (-10.0, 10.0)],
    'max_iterations': 1000,
    'global_minimum': -186.7309,
    'dimensions': 2,
}

# Multi-Dimensional Problems #
ackley_3d_config = {
    'objective': ndackley,
    'bounds': [
        (-32.768, 32.768)
    ],  # Will use a single level bound and then np.np.expand
    'max_iterations': 1000,
    'global_minimum': 0.0,
    'dimensions': 3,
}

# Multi-Dimensional Problems #
ackley_5d_config = {
    'objective': ndackley,
    'bounds': [(-32.768, 32.768)],  # Will use a single level bound and then expand
    'max_iterations': 1000,
    'global_minimum': 0.0,
    'dimensions': 5,
}

# ex8_6_2 from MINLP
ex8_6_2_config = {
    'objective': ex8_6_2,
    'bounds': [
        (0, 0),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (0, 0),
        (0, 0),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (0, 0),
        (0, 0),
        (0, 0),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
    ],
    'max_iterations': 1000,
    'global_minimum': -45,
    'dimensions': 30,
}


PROBLEMS_BY_NAME = {
    'ackley': ackley_config,
    'ackley_3d': ackley_3d_config,
    'ackley_5d': ackley_5d_config,
    'bukin_n6': bukin_n6_config,
    'cross_in_tray': cross_in_tray_config,
    'drop_wave': drop_wave_config,
    'eggholder': eggholder_config,
    'griewank': griewank_config,
    'holder_table': holder_table_config,
    'levy': levy_config,
    'levy_n13': levy_n13_config,
    'rastrigin': rastrigin_config,
    'schaffer_n2': schaffer_n2_config,
    'schaffer_n4': schaffer_n4_config,
    'schwefel': schwefel_config,
    'shubert': shubert_config,
    'ex8_6_2': ex8_6_2_config,
}
