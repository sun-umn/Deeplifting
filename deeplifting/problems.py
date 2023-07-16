# third party
import numpy as np
import torch

# first party
from deeplifting.kriging_peaks.kriging_peaks_red import kriging_peaks_red100


def ackley(x, results, trial, version='numpy'):
    """
    Function that implements the Ackley function in
    numpy or pytorch. We will use this for our deeplifting experiments.
    Note, that this version is the 2-D version only.
    """
    a = 20
    b = 0.2
    c = 2 * np.pi

    # Get x1 & x2
    x1, x2 = x.flatten()

    if version == 'numpy':
        sum_sq_term = -a * np.exp(-b * np.sqrt(0.5 * (x1**2 + x2**2)))
        cos_term = -np.exp(0.5 * (np.cos(c * x1) + np.cos(c * x2)))
        result = sum_sq_term + cos_term + a + np.exp(1)
    elif version == 'pytorch':
        sum_sq_term = -a * torch.exp(-b * torch.sqrt(0.5 * (x1**2 + x2**2)))
        cos_term = -torch.exp(0.5 * (torch.cos(c * x1) + torch.cos(c * x2)))
        result = sum_sq_term + cos_term + a + torch.exp(torch.tensor(1.0))

    else:
        raise ValueError(
            "Unknown version specified. Available " "options are 'numpy' and 'pytorch'."
        )

    # Fill in the intermediate results
    iteration = np.argmin(~np.any(np.isnan(results[trial]), axis=1))

    if isinstance(result, torch.Tensor):
        results[trial, iteration, :] = np.array(
            (
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy(),
                result.detach().cpu().numpy(),
            )
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
        result = -a * np.exp(arg1) - np.exp(arg2) + a + np.e

    elif version == 'pytorch':
        arg1 = -b * torch.sqrt(1.0 / d * torch.sum(x**2))
        arg2 = 1.0 / d * torch.sum(torch.cos(c * x))
        result = -a * torch.exp(arg1) - torch.exp(arg2) + a + np.e
    else:
        raise ValueError("Invalid implementation: choose 'numpy' or 'pytorch'")

    # Fill in the intermediate results
    iteration = np.argmin(~np.any(np.isnan(results[trial]), axis=1))

    if isinstance(result, torch.Tensor):
        x_tuple = tuple(x.detach().cpu().numpy())
        results[trial, iteration, :] = np.array(
            x_tuple + (result.detach().cpu().numpy(),)
        )

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
            (
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy(),
                result.detach().cpu().numpy(),
            )
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
                    * np.exp(np.abs(100 - np.sqrt(x1**2 + x2**2) / np.pi))
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
                    * torch.exp(
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
            (
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy(),
                result.detach().cpu().numpy(),
            )
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
            (
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy(),
                result.detach().cpu().numpy(),
            )
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, result))

    return result


def eggholder(x, results, trial, version='numpy'):
    """
    Implementation of the 2D Eggholder function.
    This function has numerous local minima and a global minimu

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
            (
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy(),
                result.detach().cpu().numpy(),
            )
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
            (
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy(),
                result.detach().cpu().numpy(),
            )
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
            * np.exp(np.abs(1 - np.sqrt(x1**2 + x2**2) / np.pi))
        )
    elif version == 'pytorch':
        result = -torch.abs(
            torch.sin(x1)
            * torch.cos(x2)
            * torch.exp(torch.abs(1 - torch.sqrt(x1**2 + x2**2) / np.pi))
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
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy(),
                result.detach().cpu().numpy(),
            )
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
            (
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy(),
                result.detach().cpu().numpy(),
            )
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
            (
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy(),
                result.detach().cpu().numpy(),
            )
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
            (
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy(),
                result.detach().cpu().numpy(),
            )
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
            (
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy(),
                result.detach().cpu().numpy(),
            )
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
            (
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy(),
                result.detach().cpu().numpy(),
            )
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
            (
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy(),
                result.detach().cpu().numpy(),
            )
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
            (
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy(),
                result.detach().cpu().numpy(),
            )
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
            (
                -np.exp(
                    -3 * ((x1 - x2) ** 2 + (x11 - x12) ** 2 + (x21 - x22) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x1 - x3) ** 2 + (x11 - x13) ** 2 + (x21 - x23) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x1 - x4) ** 2 + (x11 - x14) ** 2 + (x21 - x24) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x1 - x5) ** 2 + (x11 - x15) ** 2 + (x21 - x25) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x1 - x6) ** 2 + (x11 - x16) ** 2 + (x21 - x26) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x1 - x7) ** 2 + (x11 - x17) ** 2 + (x21 - x27) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x1 - x8) ** 2 + (x11 - x18) ** 2 + (x21 - x28) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x1 - x9) ** 2 + (x11 - x19) ** 2 + (x21 - x29) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x1 - x10) ** 2 + (x11 - x20) ** 2 + (x21 - x30) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x2 - x3) ** 2 + (x12 - x13) ** 2 + (x22 - x23) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x2 - x4) ** 2 + (x12 - x14) ** 2 + (x22 - x24) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x2 - x5) ** 2 + (x12 - x15) ** 2 + (x22 - x25) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x2 - x6) ** 2 + (x12 - x16) ** 2 + (x22 - x26) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x2 - x7) ** 2 + (x12 - x17) ** 2 + (x22 - x27) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x2 - x8) ** 2 + (x12 - x18) ** 2 + (x22 - x28) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x2 - x9) ** 2 + (x12 - x19) ** 2 + (x22 - x29) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x2 - x10) ** 2 + (x12 - x20) ** 2 + (x22 - x30) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x3 - x4) ** 2 + (x13 - x14) ** 2 + (x23 - x24) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x3 - x5) ** 2 + (x13 - x15) ** 2 + (x23 - x25) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x3 - x6) ** 2 + (x13 - x16) ** 2 + (x23 - x26) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x3 - x7) ** 2 + (x13 - x17) ** 2 + (x23 - x27) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x3 - x8) ** 2 + (x13 - x18) ** 2 + (x23 - x28) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x3 - x9) ** 2 + (x13 - x19) ** 2 + (x23 - x29) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x3 - x10) ** 2 + (x13 - x20) ** 2 + (x23 - x30) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x4 - x5) ** 2 + (x14 - x15) ** 2 + (x24 - x25) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x4 - x6) ** 2 + (x14 - x16) ** 2 + (x24 - x26) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x4 - x7) ** 2 + (x14 - x17) ** 2 + (x24 - x27) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x4 - x8) ** 2 + (x14 - x18) ** 2 + (x24 - x28) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x4 - x9) ** 2 + (x14 - x19) ** 2 + (x24 - x29) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x4 - x10) ** 2 + (x14 - x20) ** 2 + (x24 - x30) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x5 - x6) ** 2 + (x15 - x16) ** 2 + (x25 - x26) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x5 - x7) ** 2 + (x15 - x17) ** 2 + (x25 - x27) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x5 - x8) ** 2 + (x15 - x18) ** 2 + (x25 - x28) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x5 - x9) ** 2 + (x15 - x19) ** 2 + (x25 - x29) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x5 - x10) ** 2 + (x15 - x20) ** 2 + (x25 - x30) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x6 - x7) ** 2 + (x16 - x17) ** 2 + (x26 - x27) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x6 - x8) ** 2 + (x16 - x18) ** 2 + (x26 - x28) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x6 - x9) ** 2 + (x16 - x19) ** 2 + (x26 - x29) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x6 - x10) ** 2 + (x16 - x20) ** 2 + (x26 - x30) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x7 - x8) ** 2 + (x17 - x18) ** 2 + (x27 - x28) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x7 - x9) ** 2 + (x17 - x19) ** 2 + (x27 - x29) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x7 - x10) ** 2 + (x17 - x20) ** 2 + (x27 - x30) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x8 - x9) ** 2 + (x18 - x19) ** 2 + (x28 - x29) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x8 - x10) ** 2 + (x18 - x20) ** 2 + (x28 - x30) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -np.exp(
                    -3 * ((x9 - x10) ** 2 + (x19 - x20) ** 2 + (x29 - x30) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            - 45
        )

    elif version == 'pytorch':
        result = (
            (
                -torch.exp(
                    -3 * ((x1 - x2) ** 2 + (x11 - x12) ** 2 + (x21 - x22) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x1 - x3) ** 2 + (x11 - x13) ** 2 + (x21 - x23) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x1 - x4) ** 2 + (x11 - x14) ** 2 + (x21 - x24) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x1 - x5) ** 2 + (x11 - x15) ** 2 + (x21 - x25) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x1 - x6) ** 2 + (x11 - x16) ** 2 + (x21 - x26) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x1 - x7) ** 2 + (x11 - x17) ** 2 + (x21 - x27) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x1 - x8) ** 2 + (x11 - x18) ** 2 + (x21 - x28) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x1 - x9) ** 2 + (x11 - x19) ** 2 + (x21 - x29) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x1 - x10) ** 2 + (x11 - x20) ** 2 + (x21 - x30) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x2 - x3) ** 2 + (x12 - x13) ** 2 + (x22 - x23) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x2 - x4) ** 2 + (x12 - x14) ** 2 + (x22 - x24) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x2 - x5) ** 2 + (x12 - x15) ** 2 + (x22 - x25) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x2 - x6) ** 2 + (x12 - x16) ** 2 + (x22 - x26) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x2 - x7) ** 2 + (x12 - x17) ** 2 + (x22 - x27) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x2 - x8) ** 2 + (x12 - x18) ** 2 + (x22 - x28) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x2 - x9) ** 2 + (x12 - x19) ** 2 + (x22 - x29) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x2 - x10) ** 2 + (x12 - x20) ** 2 + (x22 - x30) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x3 - x4) ** 2 + (x13 - x14) ** 2 + (x23 - x24) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x3 - x5) ** 2 + (x13 - x15) ** 2 + (x23 - x25) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x3 - x6) ** 2 + (x13 - x16) ** 2 + (x23 - x26) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x3 - x7) ** 2 + (x13 - x17) ** 2 + (x23 - x27) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x3 - x8) ** 2 + (x13 - x18) ** 2 + (x23 - x28) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x3 - x9) ** 2 + (x13 - x19) ** 2 + (x23 - x29) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x3 - x10) ** 2 + (x13 - x20) ** 2 + (x23 - x30) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x4 - x5) ** 2 + (x14 - x15) ** 2 + (x24 - x25) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x4 - x6) ** 2 + (x14 - x16) ** 2 + (x24 - x26) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x4 - x7) ** 2 + (x14 - x17) ** 2 + (x24 - x27) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x4 - x8) ** 2 + (x14 - x18) ** 2 + (x24 - x28) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x4 - x9) ** 2 + (x14 - x19) ** 2 + (x24 - x29) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x4 - x10) ** 2 + (x14 - x20) ** 2 + (x24 - x30) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x5 - x6) ** 2 + (x15 - x16) ** 2 + (x25 - x26) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x5 - x7) ** 2 + (x15 - x17) ** 2 + (x25 - x27) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x5 - x8) ** 2 + (x15 - x18) ** 2 + (x25 - x28) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x5 - x9) ** 2 + (x15 - x19) ** 2 + (x25 - x29) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x5 - x10) ** 2 + (x15 - x20) ** 2 + (x25 - x30) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x6 - x7) ** 2 + (x16 - x17) ** 2 + (x26 - x27) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x6 - x8) ** 2 + (x16 - x18) ** 2 + (x26 - x28) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x6 - x9) ** 2 + (x16 - x19) ** 2 + (x26 - x29) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x6 - x10) ** 2 + (x16 - x20) ** 2 + (x26 - x30) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x7 - x8) ** 2 + (x17 - x18) ** 2 + (x27 - x28) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x7 - x9) ** 2 + (x17 - x19) ** 2 + (x27 - x29) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x7 - x10) ** 2 + (x17 - x20) ** 2 + (x27 - x30) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x8 - x9) ** 2 + (x18 - x19) ** 2 + (x28 - x29) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x8 - x10) ** 2 + (x18 - x20) ** 2 + (x28 - x30) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            + (
                -torch.exp(
                    -3 * ((x9 - x10) ** 2 + (x19 - x20) ** 2 + (x29 - x30) ** 2) ** 0.5
                    + 3
                )
                + 1
            )
            ** 2
            - 45
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
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy(),
                x3.detach().cpu().numpy(),
                x4.detach().cpu().numpy(),
                x5.detach().cpu().numpy(),
                x6.detach().cpu().numpy(),
                x7.detach().cpu().numpy(),
                x8.detach().cpu().numpy(),
                x9.detach().cpu().numpy(),
                x10.detach().cpu().numpy(),
                x11.detach().cpu().numpy(),
                x12.detach().cpu().numpy(),
                x13.detach().cpu().numpy(),
                x14.detach().cpu().numpy(),
                x15.detach().cpu().numpy(),
                x16.detach().cpu().numpy(),
                x17.detach().cpu().numpy(),
                x18.detach().cpu().numpy(),
                x19.detach().cpu().numpy(),
                x20.detach().cpu().numpy(),
                x21.detach().cpu().numpy(),
                x22.detach().cpu().numpy(),
                x23.detach().cpu().numpy(),
                x24.detach().cpu().numpy(),
                x25.detach().cpu().numpy(),
                x26.detach().cpu().numpy(),
                x27.detach().cpu().numpy(),
                x28.detach().cpu().numpy(),
                x29.detach().cpu().numpy(),
                x30.detach().cpu().numpy(),
                result.detach().cpu().numpy(),
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


def least(x, results, trial, version='numpy'):
    """
    Implementation of the Least function from the MINLP library.
    This function has a global minimum of 0.0.

    Parameters:
        x: (x1, x2, x3) this is a 3D problem
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
    x1, x2, x3 = x.flatten()
    if version == 'numpy':
        result = (
            (-x2 * np.exp(-5 * x3) + 127 - x1) ** 2
            + (-x2 * np.exp(-3 * x3) + 151 - x1) ** 2
            + (-x2 * np.exp(-x3) + 379 - x1) ** 2
            + (-x2 * np.exp(5 * x3) + 421 - x1) ** 2
            + (-x2 * np.exp(3 * x3) + 460 - x1) ** 2
            + (-x2 * np.exp(x3) + 426 - x1) ** 2
        )
    elif version == 'pytorch':
        result = (
            (-x2 * torch.exp(-5 * x3) + 127 - x1) ** 2
            + (-x2 * torch.exp(-3 * x3) + 151 - x1) ** 2
            + (-x2 * torch.exp(-x3) + 379 - x1) ** 2
            + (-x2 * torch.exp(5 * x3) + 421 - x1) ** 2
            + (-x2 * torch.exp(3 * x3) + 460 - x1) ** 2
            + (-x2 * torch.exp(x3) + 426 - x1) ** 2
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
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy(),
                x3.detach().cpu().numpy(),
                result.detach().cpu().numpy(),
            )
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, x3, result))

    return result


def ex8_1_3(x, results, trial, version='numpy'):
    """
    Implementation of the example 8-1-3 function from the MINLP library.
    This function has a global minimum of 3.0.

    Parameters:
        x: (x1, x2) this is a 3D problem
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
            (1 + x1 + x2) ** 2
            * (3 * x1**2 - 14 * x1 + 6 * x1 * x2 - 14 * x2 + 3 * x2**2 + 19)
            + 1
        ) * (
            (2 * x1 - 3 * x2) ** 2
            * (12 * x1**2 - 32 * x1 - 36 * x1 * x2 + 48 * x2 + 27 * x2**2 + 18)
            + 30
        )
    elif version == 'pytorch':
        result = (
            (1 + x1 + x2) ** 2
            * (3 * x1**2 - 14 * x1 + 6 * x1 * x2 - 14 * x2 + 3 * x2**2 + 19)
            + 1
        ) * (
            (2 * x1 - 3 * x2) ** 2
            * (12 * x1**2 - 32 * x1 - 36 * x1 * x2 + 48 * x2 + 27 * x2**2 + 18)
            + 30
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
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy(),
                result.detach().cpu().numpy(),
            )
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, result))

    return result


def ex8_1_1(x, results, trial, version='numpy'):
    """
    Implementation of the example 8-1-1 function from the MINLP library.
    This function has a global minimum of -2.0218.

    Parameters:
        x: (x1, x2) this is a 3D problem
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
        result = np.cos(x1) * np.sin(x2) - x1 / (x2**2 + 1)
    elif version == 'pytorch':
        result = torch.cos(x1) * torch.sin(x2) - x1 / (x2**2 + 1)
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

    # Fill in the intermediate results
    iteration = np.argmin(~np.any(np.isnan(results[trial]), axis=1))

    if isinstance(result, torch.Tensor):
        results[trial, iteration, :] = np.array(
            (
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy(),
                result.detach().cpu().numpy(),
            )
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, result))

    return result


def ex4_1_5(x, results, trial, version='numpy'):
    """
    Implementation of the example 4-1-5 function from the MINLP library.
    This function has a global minimum of 0.0.

    Parameters:
        x: (x1, x2) this is a 3D problem
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
        result = np.cos(x1) * np.sin(x2) - x1 / (x2**2 + 1)
    elif version == 'pytorch':
        result = torch.cos(x1) * torch.sin(x2) - x1 / (x2**2 + 1)
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

    # Fill in the intermediate results
    iteration = np.argmin(~np.any(np.isnan(results[trial]), axis=1))

    if isinstance(result, torch.Tensor):
        results[trial, iteration, :] = np.array(
            (
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy(),
                result.detach().cpu().numpy(),
            )
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, result))

    return result


def ex8_1_5(x, results, trial, version='numpy'):
    """
    Implementation of the example 8-1-5 function from the MINLP library.
    This function has a global minimum of -2.0218.

    Parameters:
        x: (x1, x2) this is a 3D problem
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
            4 * x1**2
            - 2.1 * x1**4
            + 0.333333333333333 * x1**6
            + x1 * x2
            - 4 * x2**2
            + 4 * x2**4
        )
    elif version == 'pytorch':
        result = (
            4 * x1**2
            - 2.1 * x1**4
            + 0.333333333333333 * x1**6
            + x1 * x2
            - 4 * x2**2
            + 4 * x2**4
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
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy(),
                result.detach().cpu().numpy(),
            )
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, result))

    return result


def ex8_1_4(x, results, trial, version='numpy'):
    """
    Implementation of the example 8-1-4 function from the MINLP library.
    This function has a global minimum of 0.0.

    Parameters:
        x: (x1, x2) this is a 3D problem
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
        result = 12 * x1**2 - 6.3 * x1**4 + x1**6 - 6 * x1 * x2 + 6 * x2**2
    elif version == 'pytorch':
        result = 12 * x1**2 - 6.3 * x1**4 + x1**6 - 6 * x1 * x2 + 6 * x2**2
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

    # Fill in the intermediate results
    iteration = np.argmin(~np.any(np.isnan(results[trial]), axis=1))

    if isinstance(result, torch.Tensor):
        results[trial, iteration, :] = np.array(
            (
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy(),
                result.detach().cpu().numpy(),
            )
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, result))

    return result


def ex8_1_6(x, results, trial, version='numpy'):
    """
    Implementation of the example 8-1-6 function from the MINLP library.
    This function has a global minimum of -10.0886. This
    was found by SCIP

    Parameters:
        x: (x1, x2) this is a 3D problem
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
            -1 / ((-4 + x1) ** 2 + (-4 + x2) ** 2 + 0.1)
            - 1 / ((-1 + x1) ** 2 + (-1 + x2) ** 2 + 0.2)
            - 1 / ((-8 + x1) ** 2 + (-8 + x2) ** 2 + 0.2)
        )
    elif version == 'pytorch':
        result = (
            -1 / ((-4 + x1) ** 2 + (-4 + x2) ** 2 + 0.1)
            - 1 / ((-1 + x1) ** 2 + (-1 + x2) ** 2 + 0.2)
            - 1 / ((-8 + x1) ** 2 + (-8 + x2) ** 2 + 0.2)
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
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy(),
                result.detach().cpu().numpy(),
            )
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, result))

    return result


def kriging_peaks_red010(x, results, trial, version='numpy'):
    """
    Implementation of the kriging peaks red-010 function from the MINLP library.
    This function has a global minimum of 0.2911. This
    was found by SCIP

    Parameters:
        x: (x1, x2) this is a 3D problem
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
            -1.86360571672641
            * (
                (
                    0.14220168012508
                    * (
                        2.23606797749979
                        * np.sqrt(
                            64.2558879895505
                            * (0.194030125231034 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (-0.404841797459896 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            64.2558879895505
                            * (0.194030125231034 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (-0.404841797459896 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.14220168012508
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        64.2558879895505
                        * (0.194030125231034 - 0.166666666666667 * x1) ** 2
                        + 0.451453304154821
                        * (-0.404841797459896 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0468045532937668
                    * (
                        2.23606797749979
                        * np.sqrt(
                            64.2558879895505
                            * (0.268324435572052 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (-0.220830234095203 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            64.2558879895505
                            * (0.268324435572052 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (-0.220830234095203 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0468045532937668
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        64.2558879895505
                        * (0.268324435572052 - 0.166666666666667 * x1) ** 2
                        + 0.451453304154821
                        * (-0.220830234095203 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.436827958615331
                    * (
                        2.23606797749979
                        * np.sqrt(
                            64.2558879895505
                            * (-0.0505822037784427 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (0.190399166851636 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            64.2558879895505
                            * (-0.0505822037784427 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (0.190399166851636 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.436827958615331
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        64.2558879895505
                        * (-0.0505822037784427 - 0.166666666666667 * x1) ** 2
                        + 0.451453304154821
                        * (0.190399166851636 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0405780260973915
                    * (
                        2.23606797749979
                        * np.sqrt(
                            64.2558879895505
                            * (0.335653234585858 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (0.319274707641782 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            64.2558879895505
                            * (0.335653234585858 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (0.319274707641782 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0405780260973915
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        64.2558879895505
                        * (0.335653234585858 - 0.166666666666667 * x1) ** 2
                        + 0.451453304154821
                        * (0.319274707641782 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0383717635737363
                    * (
                        2.23606797749979
                        * np.sqrt(
                            64.2558879895505
                            * (0.416115262788179 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (-0.0829399801155143 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            64.2558879895505
                            * (0.416115262788179 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (-0.0829399801155143 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0383717635737363
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        64.2558879895505
                        * (0.416115262788179 - 0.166666666666667 * x1) ** 2
                        + 0.451453304154821
                        * (-0.0829399801155143 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.226121620987036
                    * (
                        2.23606797749979
                        * np.sqrt(
                            64.2558879895505
                            * (-0.324344059222606 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (0.0775173811415889 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            64.2558879895505
                            * (-0.324344059222606 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (0.0775173811415889 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.226121620987036
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        64.2558879895505
                        * (-0.324344059222606 - 0.166666666666667 * x1) ** 2
                        + 0.451453304154821
                        * (0.0775173811415889 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.055301550115541
                    * (
                        2.23606797749979
                        * np.sqrt(
                            64.2558879895505
                            * (-0.289414019566429 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (-0.337636568589209 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            64.2558879895505
                            * (-0.289414019566429 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (-0.337636568589209 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.055301550115541
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        64.2558879895505
                        * (-0.289414019566429 - 0.166666666666667 * x1) ** 2
                        + 0.451453304154821
                        * (-0.337636568589209 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.220359443423157
                    * (
                        2.23606797749979
                        * np.sqrt(
                            64.2558879895505
                            * (-0.120888900038076 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (-0.125103261913789 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            64.2558879895505
                            * (-0.120888900038076 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (-0.125103261913789 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.220359443423157
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        64.2558879895505
                        * (-0.120888900038076 - 0.166666666666667 * x1) ** 2
                        + 0.451453304154821
                        * (-0.125103261913789 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0439612252066741
                    * (
                        2.23606797749979
                        * np.sqrt(
                            64.2558879895505
                            * (0.0826904704977199 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (0.458984274331446 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            64.2558879895505
                            * (0.0826904704977199 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (0.458984274331446 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0439612252066741
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        64.2558879895505
                        * (0.0826904704977199 - 0.166666666666667 * x1) ** 2
                        + 0.451453304154821
                        * (0.458984274331446 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0560980731767207
                    * (
                        2.23606797749979
                        * np.sqrt(
                            64.2558879895505
                            * (-0.435192808594306 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (0.220400711043696 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            64.2558879895505
                            * (-0.435192808594306 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (0.220400711043696 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0560980731767207
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        64.2558879895505
                        * (-0.435192808594306 - 0.166666666666667 * x1) ** 2
                        + 0.451453304154821
                        * (0.220400711043696 - 0.166666666666667 * x2) ** 2
                    )
                )
            )
            + 0.727377686265491
        )
    elif version == 'pytorch':
        result = (
            -1.86360571672641
            * (
                (
                    0.14220168012508
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            64.2558879895505
                            * (0.194030125231034 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (-0.404841797459896 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            64.2558879895505
                            * (0.194030125231034 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (-0.404841797459896 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.14220168012508
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        64.2558879895505
                        * (0.194030125231034 - 0.166666666666667 * x1) ** 2
                        + 0.451453304154821
                        * (-0.404841797459896 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0468045532937668
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            64.2558879895505
                            * (0.268324435572052 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (-0.220830234095203 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            64.2558879895505
                            * (0.268324435572052 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (-0.220830234095203 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0468045532937668
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        64.2558879895505
                        * (0.268324435572052 - 0.166666666666667 * x1) ** 2
                        + 0.451453304154821
                        * (-0.220830234095203 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.436827958615331
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            64.2558879895505
                            * (-0.0505822037784427 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (0.190399166851636 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            64.2558879895505
                            * (-0.0505822037784427 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (0.190399166851636 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.436827958615331
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        64.2558879895505
                        * (-0.0505822037784427 - 0.166666666666667 * x1) ** 2
                        + 0.451453304154821
                        * (0.190399166851636 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0405780260973915
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            64.2558879895505
                            * (0.335653234585858 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (0.319274707641782 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            64.2558879895505
                            * (0.335653234585858 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (0.319274707641782 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0405780260973915
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        64.2558879895505
                        * (0.335653234585858 - 0.166666666666667 * x1) ** 2
                        + 0.451453304154821
                        * (0.319274707641782 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0383717635737363
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            64.2558879895505
                            * (0.416115262788179 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (-0.0829399801155143 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            64.2558879895505
                            * (0.416115262788179 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (-0.0829399801155143 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0383717635737363
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        64.2558879895505
                        * (0.416115262788179 - 0.166666666666667 * x1) ** 2
                        + 0.451453304154821
                        * (-0.0829399801155143 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.226121620987036
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            64.2558879895505
                            * (-0.324344059222606 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (0.0775173811415889 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            64.2558879895505
                            * (-0.324344059222606 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (0.0775173811415889 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.226121620987036
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        64.2558879895505
                        * (-0.324344059222606 - 0.166666666666667 * x1) ** 2
                        + 0.451453304154821
                        * (0.0775173811415889 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.055301550115541
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            64.2558879895505
                            * (-0.289414019566429 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (-0.337636568589209 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            64.2558879895505
                            * (-0.289414019566429 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (-0.337636568589209 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.055301550115541
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        64.2558879895505
                        * (-0.289414019566429 - 0.166666666666667 * x1) ** 2
                        + 0.451453304154821
                        * (-0.337636568589209 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.220359443423157
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            64.2558879895505
                            * (-0.120888900038076 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (-0.125103261913789 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            64.2558879895505
                            * (-0.120888900038076 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (-0.125103261913789 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.220359443423157
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        64.2558879895505
                        * (-0.120888900038076 - 0.166666666666667 * x1) ** 2
                        + 0.451453304154821
                        * (-0.125103261913789 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0439612252066741
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            64.2558879895505
                            * (0.0826904704977199 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (0.458984274331446 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            64.2558879895505
                            * (0.0826904704977199 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (0.458984274331446 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0439612252066741
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        64.2558879895505
                        * (0.0826904704977199 - 0.166666666666667 * x1) ** 2
                        + 0.451453304154821
                        * (0.458984274331446 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0560980731767207
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            64.2558879895505
                            * (-0.435192808594306 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (0.220400711043696 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            64.2558879895505
                            * (-0.435192808594306 - 0.166666666666667 * x1) ** 2
                            + 0.451453304154821
                            * (0.220400711043696 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0560980731767207
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        64.2558879895505
                        * (-0.435192808594306 - 0.166666666666667 * x1) ** 2
                        + 0.451453304154821
                        * (0.220400711043696 - 0.166666666666667 * x2) ** 2
                    )
                )
            )
            + 0.727377686265491
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
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy(),
                result.detach().cpu().numpy(),
            )
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, result))

    return result


def kriging_peaks_red020(x, results, trial, version='numpy'):
    """
    Implementation of the kriging peaks red-020 function from the MINLP library.
    This function has a global minimum of 0.3724. This
    was found by SCIP

    Parameters:
        x: (x1, x2) this is a 3D problem
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
            -1.60096340792774
            * (
                (
                    0.00793029005459249
                    * (
                        2.23606797749979
                        * np.sqrt(
                            0.73427818977281
                            * (-0.296051567570993 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.332922698952783 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (-0.296051567570993 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.332922698952783 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.00793029005459249
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        0.73427818977281
                        * (-0.296051567570993 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (-0.332922698952783 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0817257968484589
                    * (
                        2.23606797749979
                        * np.sqrt(
                            0.73427818977281
                            * (-0.114134289781611 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.210629474506774 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (-0.114134289781611 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.210629474506774 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0817257968484589
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        0.73427818977281
                        * (-0.114134289781611 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (0.210629474506774 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0195883345333713
                    * (
                        2.23606797749979
                        * np.sqrt(
                            0.73427818977281
                            * (0.236967756917555 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.252652716266839 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (0.236967756917555 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.252652716266839 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0195883345333713
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        0.73427818977281
                        * (0.236967756917555 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (0.252652716266839 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0291783688013098
                    * (
                        2.23606797749979
                        * np.sqrt(
                            0.73427818977281
                            * (0.152741639941018 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.419722496392701 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (0.152741639941018 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.419722496392701 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0291783688013098
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        0.73427818977281
                        * (0.152741639941018 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (-0.419722496392701 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.00325119550262763
                    * (
                        2.23606797749979
                        * np.sqrt(
                            0.73427818977281
                            * (-0.226947162875829 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.267754124173417 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (-0.226947162875829 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.267754124173417 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.00325119550262763
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        0.73427818977281
                        * (-0.226947162875829 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (-0.267754124173417 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0119653901186555
                    * (
                        2.23606797749979
                        * np.sqrt(
                            0.73427818977281
                            * (-0.304487512086195 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.32234538166963 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (-0.304487512086195 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.32234538166963 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0119653901186555
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        0.73427818977281
                        * (-0.304487512086195 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (0.32234538166963 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0122437988218235
                    * (
                        2.23606797749979
                        * np.sqrt(
                            0.73427818977281
                            * (0.290016403526876 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.163073561881453 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (0.290016403526876 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.163073561881453 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0122437988218235
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        0.73427818977281
                        * (0.290016403526876 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (0.163073561881453 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0598894187216831
                    * (
                        2.23606797749979
                        * np.sqrt(
                            0.73427818977281
                            * (-0.028422744107951 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.114220818191337 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (-0.028422744107951 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.114220818191337 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0598894187216831
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        0.73427818977281
                        * (-0.028422744107951 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (-0.114220818191337 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.00735040755529198
                    * (
                        2.23606797749979
                        * np.sqrt(
                            0.73427818977281
                            * (0.432437642614655 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.0734305425446243 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (0.432437642614655 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.0734305425446243 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.00735040755529198
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        0.73427818977281
                        * (0.432437642614655 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (-0.0734305425446243 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0250062493467898
                    * (
                        2.23606797749979
                        * np.sqrt(
                            0.73427818977281
                            * (0.00988872147054509 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.130133547648043 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (0.00988872147054509 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.130133547648043 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0250062493467898
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        0.73427818977281
                        * (0.00988872147054509 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (0.130133547648043 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0132702698034036
                    * (
                        2.23606797749979
                        * np.sqrt(
                            0.73427818977281
                            * (-0.486352317260064 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.0292101825999737 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (-0.486352317260064 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.0292101825999737 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0132702698034036
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        0.73427818977281
                        * (-0.486352317260064 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (-0.0292101825999737 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.00978837984043915
                    * (
                        2.23606797749979
                        * np.sqrt(
                            0.73427818977281
                            * (-0.442402828611114 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.381430783672409 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (-0.442402828611114 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.381430783672409 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.00978837984043915
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        0.73427818977281
                        * (-0.442402828611114 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (-0.381430783672409 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0616534293016391
                    * (
                        2.23606797749979
                        * np.sqrt(
                            0.73427818977281
                            * (0.0630833526449023 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.161803864149878 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (0.0630833526449023 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.161803864149878 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0616534293016391
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        0.73427818977281
                        * (0.0630833526449023 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (-0.161803864149878 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.00806430230107368
                    * (
                        2.23606797749979
                        * np.sqrt(
                            0.73427818977281
                            * (0.339284400175223 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.240356107705697 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (0.339284400175223 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.240356107705697 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.00806430230107368
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        0.73427818977281
                        * (0.339284400175223 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (-0.240356107705697 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0100201036166177
                    * (
                        2.23606797749979
                        * np.sqrt(
                            0.73427818977281
                            * (0.474011769354471 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.473217136809233 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (0.474011769354471 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.473217136809233 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0100201036166177
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        0.73427818977281
                        * (0.474011769354471 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (-0.473217136809233 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0159637159138592
                    * (
                        2.23606797749979
                        * np.sqrt(
                            0.73427818977281
                            * (-0.38713502516494 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.423555128783197 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (-0.38713502516494 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.423555128783197 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0159637159138592
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        0.73427818977281
                        * (-0.38713502516494 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (0.423555128783197 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0740730531953568
                    * (
                        2.23606797749979
                        * np.sqrt(
                            0.73427818977281
                            * (-0.0644378540136621 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.365720321769559 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (-0.0644378540136621 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.365720321769559 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0740730531953568
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        0.73427818977281
                        * (-0.0644378540136621 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (0.365720321769559 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.00774573140270751
                    * (
                        2.23606797749979
                        * np.sqrt(
                            0.73427818977281
                            * (0.13794971820171 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.467908882200314 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (0.13794971820171 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.467908882200314 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.00774573140270751
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        0.73427818977281
                        * (0.13794971820171 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (0.467908882200314 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0865462302102454
                    * (
                        2.23606797749979
                        * np.sqrt(
                            0.73427818977281
                            * (-0.198460020780232 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.0418145422582042 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (-0.198460020780232 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.0418145422582042 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0865462302102454
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        0.73427818977281
                        * (-0.198460020780232 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (0.0418145422582042 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.00316326615674027
                    * (
                        2.23606797749979
                        * np.sqrt(
                            0.73427818977281
                            * (0.383470054705633 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.0926285234607758 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (0.383470054705633 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.0926285234607758 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.00316326615674027
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        0.73427818977281
                        * (0.383470054705633 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (0.0926285234607758 - 0.166666666666667 * x2) ** 2
                    )
                )
            )
            + 0.50206779640133
        )
    elif version == 'pytorch':
        result = (
            -1.60096340792774
            * (
                (
                    0.00793029005459249
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            0.73427818977281
                            * (-0.296051567570993 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.332922698952783 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (-0.296051567570993 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.332922698952783 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.00793029005459249
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        0.73427818977281
                        * (-0.296051567570993 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (-0.332922698952783 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0817257968484589
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            0.73427818977281
                            * (-0.114134289781611 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.210629474506774 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (-0.114134289781611 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.210629474506774 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0817257968484589
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        0.73427818977281
                        * (-0.114134289781611 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (0.210629474506774 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0195883345333713
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            0.73427818977281
                            * (0.236967756917555 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.252652716266839 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (0.236967756917555 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.252652716266839 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0195883345333713
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        0.73427818977281
                        * (0.236967756917555 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (0.252652716266839 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0291783688013098
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            0.73427818977281
                            * (0.152741639941018 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.419722496392701 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (0.152741639941018 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.419722496392701 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0291783688013098
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        0.73427818977281
                        * (0.152741639941018 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (-0.419722496392701 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.00325119550262763
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            0.73427818977281
                            * (-0.226947162875829 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.267754124173417 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (-0.226947162875829 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.267754124173417 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.00325119550262763
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        0.73427818977281
                        * (-0.226947162875829 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (-0.267754124173417 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0119653901186555
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            0.73427818977281
                            * (-0.304487512086195 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.32234538166963 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (-0.304487512086195 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.32234538166963 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0119653901186555
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        0.73427818977281
                        * (-0.304487512086195 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (0.32234538166963 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0122437988218235
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            0.73427818977281
                            * (0.290016403526876 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.163073561881453 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (0.290016403526876 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.163073561881453 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0122437988218235
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        0.73427818977281
                        * (0.290016403526876 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (0.163073561881453 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0598894187216831
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            0.73427818977281
                            * (-0.028422744107951 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.114220818191337 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (-0.028422744107951 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.114220818191337 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0598894187216831
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        0.73427818977281
                        * (-0.028422744107951 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (-0.114220818191337 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.00735040755529198
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            0.73427818977281
                            * (0.432437642614655 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.0734305425446243 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (0.432437642614655 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.0734305425446243 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.00735040755529198
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        0.73427818977281
                        * (0.432437642614655 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (-0.0734305425446243 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0250062493467898
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            0.73427818977281
                            * (0.00988872147054509 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.130133547648043 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (0.00988872147054509 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.130133547648043 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0250062493467898
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        0.73427818977281
                        * (0.00988872147054509 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (0.130133547648043 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0132702698034036
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            0.73427818977281
                            * (-0.486352317260064 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.0292101825999737 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (-0.486352317260064 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.0292101825999737 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0132702698034036
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        0.73427818977281
                        * (-0.486352317260064 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (-0.0292101825999737 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.00978837984043915
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            0.73427818977281
                            * (-0.442402828611114 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.381430783672409 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (-0.442402828611114 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.381430783672409 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.00978837984043915
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        0.73427818977281
                        * (-0.442402828611114 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (-0.381430783672409 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0616534293016391
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            0.73427818977281
                            * (0.0630833526449023 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.161803864149878 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (0.0630833526449023 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.161803864149878 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0616534293016391
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        0.73427818977281
                        * (0.0630833526449023 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (-0.161803864149878 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.00806430230107368
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            0.73427818977281
                            * (0.339284400175223 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.240356107705697 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (0.339284400175223 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.240356107705697 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.00806430230107368
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        0.73427818977281
                        * (0.339284400175223 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (-0.240356107705697 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0100201036166177
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            0.73427818977281
                            * (0.474011769354471 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.473217136809233 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (0.474011769354471 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (-0.473217136809233 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0100201036166177
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        0.73427818977281
                        * (0.474011769354471 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (-0.473217136809233 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0159637159138592
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            0.73427818977281
                            * (-0.38713502516494 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.423555128783197 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (-0.38713502516494 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.423555128783197 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0159637159138592
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        0.73427818977281
                        * (-0.38713502516494 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (0.423555128783197 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0740730531953568
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            0.73427818977281
                            * (-0.0644378540136621 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.365720321769559 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (-0.0644378540136621 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.365720321769559 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0740730531953568
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        0.73427818977281
                        * (-0.0644378540136621 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (0.365720321769559 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.00774573140270751
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            0.73427818977281
                            * (0.13794971820171 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.467908882200314 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (0.13794971820171 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.467908882200314 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.00774573140270751
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        0.73427818977281
                        * (0.13794971820171 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (0.467908882200314 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0865462302102454
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            0.73427818977281
                            * (-0.198460020780232 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.0418145422582042 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (-0.198460020780232 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.0418145422582042 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0865462302102454
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        0.73427818977281
                        * (-0.198460020780232 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (0.0418145422582042 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.00316326615674027
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            0.73427818977281
                            * (0.383470054705633 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.0926285234607758 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            0.73427818977281
                            * (0.383470054705633 - 0.166666666666667 * x1) ** 2
                            + 9.84239329440621
                            * (0.0926285234607758 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.00316326615674027
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        0.73427818977281
                        * (0.383470054705633 - 0.166666666666667 * x1) ** 2
                        + 9.84239329440621
                        * (0.0926285234607758 - 0.166666666666667 * x2) ** 2
                    )
                )
            )
            + 0.50206779640133
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
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy(),
                result.detach().cpu().numpy(),
            )
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, result))

    return result


def kriging_peaks_red030(x, results, trial, version='numpy'):
    """
    Implementation of the kriging peaks red-030 from the MINLP library.
    This function has a global minimum of -1.5886. This
    was found by ANTIGONE

    Parameters:
        x: (x1, x2) this is a 2D problem
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
            -1.70327473357547
            * (
                (
                    0.0377862851255559
                    * (
                        2.23606797749979
                        * np.sqrt(
                            28.8031207707063
                            * (-0.388770057297588 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.0954107225698604 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (-0.388770057297588 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.0954107225698604 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0377862851255559
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        28.8031207707063
                        * (-0.388770057297588 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (0.0954107225698604 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0109802834996097
                    * (
                        2.23606797749979
                        * np.sqrt(
                            28.8031207707063
                            * (0.363858827199432 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.183898783136516 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (0.363858827199432 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.183898783136516 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0109802834996097
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        28.8031207707063
                        * (0.363858827199432 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (0.183898783136516 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.00247095708560322
                    * (
                        2.23606797749979
                        * np.sqrt(
                            28.8031207707063
                            * (-0.404809280490364 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.487881605457913 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (-0.404809280490364 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.487881605457913 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.00247095708560322
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        28.8031207707063
                        * (-0.404809280490364 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (0.487881605457913 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.00863137678697887
                    * (
                        2.23606797749979
                        * np.sqrt(
                            28.8031207707063
                            * (0.230646280982898 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.258257413555802 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (0.230646280982898 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.258257413555802 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.00863137678697887
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        28.8031207707063
                        * (0.230646280982898 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (-0.258257413555802 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0426402643306017
                    * (
                        2.23606797749979
                        * np.sqrt(
                            28.8031207707063
                            * (-0.179761212621321 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.281703064648647 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (-0.179761212621321 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.281703064648647 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0426402643306017
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        28.8031207707063
                        * (-0.179761212621321 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (-0.281703064648647 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0329541013851422
                    * (
                        2.23606797749979
                        * np.sqrt(
                            28.8031207707063
                            * (0.37059458573363 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.301085894330831 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (0.37059458573363 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.301085894330831 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0329541013851422
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        28.8031207707063
                        * (0.37059458573363 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (0.301085894330831 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0101854174405717
                    * (
                        2.23606797749979
                        * np.sqrt(
                            28.8031207707063
                            * (-0.458435854020719 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.159793606686887 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (-0.458435854020719 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.159793606686887 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0101854174405717
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        28.8031207707063
                        * (-0.458435854020719 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (0.159793606686887 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.238112023839818
                    * (
                        2.23606797749979
                        * np.sqrt(
                            28.8031207707063
                            * (0.301433475166551 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.0976077173072188 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (0.301433475166551 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.0976077173072188 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.238112023839818
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        28.8031207707063
                        * (0.301433475166551 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (-0.0976077173072188 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.137616498009308
                    * (
                        2.23606797749979
                        * np.sqrt(
                            28.8031207707063
                            * (-0.0808655453781928 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.0209743388744684 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (-0.0808655453781928 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.0209743388744684 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.137616498009308
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        28.8031207707063
                        * (-0.0808655453781928 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (0.0209743388744684 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0197537725332181
                    * (
                        2.23606797749979
                        * np.sqrt(
                            28.8031207707063
                            * (0.417201570287689 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.344577905208669 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (0.417201570287689 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.344577905208669 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0197537725332181
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        28.8031207707063
                        * (0.417201570287689 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (0.344577905208669 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.387338449566282
                    * (
                        2.23606797749979
                        * np.sqrt(
                            28.8031207707063
                            * (-0.0474495115676595 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.0345290896581875 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (-0.0474495115676595 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.0345290896581875 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.387338449566282
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        28.8031207707063
                        * (-0.0474495115676595 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (-0.0345290896581875 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0354836253102756
                    * (
                        2.23606797749979
                        * np.sqrt(
                            28.8031207707063
                            * (-0.118913851966317 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.471558913298767 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (-0.118913851966317 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.471558913298767 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0354836253102756
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        28.8031207707063
                        * (-0.118913851966317 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (-0.471558913298767 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.354793561793949
                    * (
                        2.23606797749979
                        * np.sqrt(
                            28.8031207707063
                            * (0.164445710060874 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.215687166073911 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (0.164445710060874 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.215687166073911 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.354793561793949
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        28.8031207707063
                        * (0.164445710060874 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (0.215687166073911 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.235619142707833
                    * (
                        2.23606797749979
                        * np.sqrt(
                            28.8031207707063
                            * (-0.14216290604526 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.057803528332154 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (-0.14216290604526 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.057803528332154 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.235619142707833
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        28.8031207707063
                        * (-0.14216290604526 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (0.057803528332154 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.180184382909917
                    * (
                        2.23606797749979
                        * np.sqrt(
                            28.8031207707063
                            * (-0.209262561689916 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.284795849934872 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (-0.209262561689916 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.284795849934872 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.180184382909917
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        28.8031207707063
                        * (-0.209262561689916 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (0.284795849934872 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0187216593054939
                    * (
                        2.23606797749979
                        * np.sqrt(
                            28.8031207707063
                            * (-0.315454840517351 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.428235127551464 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (-0.315454840517351 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.428235127551464 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0187216593054939
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        28.8031207707063
                        * (-0.315454840517351 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (-0.428235127551464 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0482917132480993
                    * (
                        2.23606797749979
                        * np.sqrt(
                            28.8031207707063
                            * (0.0227175523502599 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.162641033566093 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (0.0227175523502599 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.162641033566093 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0482917132480993
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        28.8031207707063
                        * (0.0227175523502599 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (-0.162641033566093 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.030643742948902
                    * (
                        2.23606797749979
                        * np.sqrt(
                            28.8031207707063
                            * (-0.254371702106301 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.393221856773031 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (-0.254371702106301 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.393221856773031 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.030643742948902
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        28.8031207707063
                        * (-0.254371702106301 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (-0.393221856773031 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.00421991578602876
                    * (
                        2.23606797749979
                        * np.sqrt(
                            28.8031207707063
                            * (0.448138629452501 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.117836115520705 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (0.448138629452501 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.117836115520705 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.00421991578602876
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        28.8031207707063
                        * (0.448138629452501 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (-0.117836115520705 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.6484753341733
                    * (
                        2.23606797749979
                        * np.sqrt(
                            28.8031207707063
                            * (0.0565108353075588 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.227817045121909 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (0.0565108353075588 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.227817045121909 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.6484753341733
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        28.8031207707063
                        * (0.0565108353075588 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (-0.227817045121909 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.016576230553694
                    * (
                        2.23606797749979
                        * np.sqrt(
                            28.8031207707063
                            * (-0.477619818593827 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.110730424339593 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (-0.477619818593827 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.110730424339593 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.016576230553694
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        28.8031207707063
                        * (-0.477619818593827 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (0.110730424339593 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0274666267179964
                    * (
                        2.23606797749979
                        * np.sqrt(
                            28.8031207707063
                            * (-0.00209482449703213 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.439795553204386 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (-0.00209482449703213 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.439795553204386 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0274666267179964
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        28.8031207707063
                        * (-0.00209482449703213 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (-0.439795553204386 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0633057832782454
                    * (
                        2.23606797749979
                        * np.sqrt(
                            28.8031207707063
                            * (0.292485426296965 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.348061674089151 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (0.292485426296965 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.348061674089151 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0633057832782454
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        28.8031207707063
                        * (0.292485426296965 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (-0.348061674089151 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.104894965105221
                    * (
                        2.23606797749979
                        * np.sqrt(
                            28.8031207707063
                            * (0.170655556838736 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.376936461235205 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (0.170655556838736 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.376936461235205 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.104894965105221
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        28.8031207707063
                        * (0.170655556838736 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (0.376936461235205 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0254209661646174
                    * (
                        2.23606797749979
                        * np.sqrt(
                            28.8031207707063
                            * (-0.292026817763947 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.168826707824519 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (-0.292026817763947 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.168826707824519 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0254209661646174
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        28.8031207707063
                        * (-0.292026817763947 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (-0.168826707824519 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.00365601748771598
                    * (
                        2.23606797749979
                        * np.sqrt(
                            28.8031207707063
                            * (0.466799636494498 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.403623280335378 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (0.466799636494498 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.403623280335378 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.00365601748771598
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        28.8031207707063
                        * (0.466799636494498 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (0.403623280335378 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.109257824853306
                    * (
                        2.23606797749979
                        * np.sqrt(
                            28.8031207707063
                            * (0.23547573937596 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.250945430575571 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (0.23547573937596 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.250945430575571 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.109257824853306
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        28.8031207707063
                        * (0.23547573937596 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (0.250945430575571 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.480286507430443
                    * (
                        2.23606797749979
                        * np.sqrt(
                            28.8031207707063
                            * (0.110566185432721 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.301154017333651 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (0.110566185432721 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.301154017333651 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.480286507430443
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        28.8031207707063
                        * (0.110566185432721 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (-0.301154017333651 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.105196793968471
                    * (
                        2.23606797749979
                        * np.sqrt(
                            28.8031207707063
                            * (-0.358055339376858 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.0107111341783262 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (-0.358055339376858 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.0107111341783262 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.105196793968471
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        28.8031207707063
                        * (-0.358055339376858 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (-0.0107111341783262 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0375716901420036
                    * (
                        2.23606797749979
                        * np.sqrt(
                            28.8031207707063
                            * (0.0785585414434151 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.462551216897219 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (0.0785585414434151 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.462551216897219 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0375716901420036
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        28.8031207707063
                        * (0.0785585414434151 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (0.462551216897219 - 0.166666666666667 * x2) ** 2
                    )
                )
            )
            - 0.104337166479966
        )
    elif version == 'pytorch':
        result = (
            -1.70327473357547
            * (
                (
                    0.0377862851255559
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            28.8031207707063
                            * (-0.388770057297588 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.0954107225698604 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (-0.388770057297588 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.0954107225698604 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0377862851255559
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        28.8031207707063
                        * (-0.388770057297588 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (0.0954107225698604 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0109802834996097
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            28.8031207707063
                            * (0.363858827199432 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.183898783136516 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (0.363858827199432 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.183898783136516 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0109802834996097
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        28.8031207707063
                        * (0.363858827199432 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (0.183898783136516 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.00247095708560322
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            28.8031207707063
                            * (-0.404809280490364 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.487881605457913 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (-0.404809280490364 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.487881605457913 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.00247095708560322
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        28.8031207707063
                        * (-0.404809280490364 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (0.487881605457913 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.00863137678697887
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            28.8031207707063
                            * (0.230646280982898 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.258257413555802 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (0.230646280982898 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.258257413555802 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.00863137678697887
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        28.8031207707063
                        * (0.230646280982898 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (-0.258257413555802 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0426402643306017
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            28.8031207707063
                            * (-0.179761212621321 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.281703064648647 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (-0.179761212621321 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.281703064648647 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0426402643306017
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        28.8031207707063
                        * (-0.179761212621321 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (-0.281703064648647 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0329541013851422
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            28.8031207707063
                            * (0.37059458573363 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.301085894330831 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (0.37059458573363 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.301085894330831 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0329541013851422
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        28.8031207707063
                        * (0.37059458573363 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (0.301085894330831 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0101854174405717
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            28.8031207707063
                            * (-0.458435854020719 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.159793606686887 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (-0.458435854020719 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.159793606686887 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0101854174405717
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        28.8031207707063
                        * (-0.458435854020719 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (0.159793606686887 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.238112023839818
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            28.8031207707063
                            * (0.301433475166551 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.0976077173072188 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (0.301433475166551 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.0976077173072188 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.238112023839818
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        28.8031207707063
                        * (0.301433475166551 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (-0.0976077173072188 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.137616498009308
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            28.8031207707063
                            * (-0.0808655453781928 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.0209743388744684 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (-0.0808655453781928 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.0209743388744684 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.137616498009308
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        28.8031207707063
                        * (-0.0808655453781928 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (0.0209743388744684 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0197537725332181
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            28.8031207707063
                            * (0.417201570287689 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.344577905208669 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (0.417201570287689 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.344577905208669 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0197537725332181
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        28.8031207707063
                        * (0.417201570287689 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (0.344577905208669 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.387338449566282
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            28.8031207707063
                            * (-0.0474495115676595 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.0345290896581875 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (-0.0474495115676595 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.0345290896581875 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.387338449566282
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        28.8031207707063
                        * (-0.0474495115676595 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (-0.0345290896581875 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0354836253102756
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            28.8031207707063
                            * (-0.118913851966317 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.471558913298767 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (-0.118913851966317 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.471558913298767 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0354836253102756
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        28.8031207707063
                        * (-0.118913851966317 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (-0.471558913298767 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.354793561793949
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            28.8031207707063
                            * (0.164445710060874 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.215687166073911 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (0.164445710060874 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.215687166073911 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.354793561793949
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        28.8031207707063
                        * (0.164445710060874 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (0.215687166073911 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.235619142707833
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            28.8031207707063
                            * (-0.14216290604526 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.057803528332154 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (-0.14216290604526 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.057803528332154 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.235619142707833
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        28.8031207707063
                        * (-0.14216290604526 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (0.057803528332154 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.180184382909917
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            28.8031207707063
                            * (-0.209262561689916 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.284795849934872 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (-0.209262561689916 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.284795849934872 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.180184382909917
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        28.8031207707063
                        * (-0.209262561689916 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (0.284795849934872 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0187216593054939
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            28.8031207707063
                            * (-0.315454840517351 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.428235127551464 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (-0.315454840517351 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.428235127551464 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0187216593054939
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        28.8031207707063
                        * (-0.315454840517351 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (-0.428235127551464 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0482917132480993
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            28.8031207707063
                            * (0.0227175523502599 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.162641033566093 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (0.0227175523502599 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.162641033566093 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0482917132480993
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        28.8031207707063
                        * (0.0227175523502599 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (-0.162641033566093 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.030643742948902
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            28.8031207707063
                            * (-0.254371702106301 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.393221856773031 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (-0.254371702106301 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.393221856773031 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.030643742948902
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        28.8031207707063
                        * (-0.254371702106301 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (-0.393221856773031 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.00421991578602876
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            28.8031207707063
                            * (0.448138629452501 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.117836115520705 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (0.448138629452501 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.117836115520705 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.00421991578602876
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        28.8031207707063
                        * (0.448138629452501 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (-0.117836115520705 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.6484753341733
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            28.8031207707063
                            * (0.0565108353075588 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.227817045121909 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (0.0565108353075588 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.227817045121909 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.6484753341733
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        28.8031207707063
                        * (0.0565108353075588 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (-0.227817045121909 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.016576230553694
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            28.8031207707063
                            * (-0.477619818593827 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.110730424339593 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (-0.477619818593827 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.110730424339593 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.016576230553694
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        28.8031207707063
                        * (-0.477619818593827 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (0.110730424339593 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0274666267179964
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            28.8031207707063
                            * (-0.00209482449703213 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.439795553204386 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (-0.00209482449703213 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.439795553204386 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0274666267179964
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        28.8031207707063
                        * (-0.00209482449703213 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (-0.439795553204386 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0633057832782454
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            28.8031207707063
                            * (0.292485426296965 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.348061674089151 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (0.292485426296965 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.348061674089151 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0633057832782454
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        28.8031207707063
                        * (0.292485426296965 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (-0.348061674089151 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.104894965105221
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            28.8031207707063
                            * (0.170655556838736 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.376936461235205 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (0.170655556838736 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.376936461235205 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.104894965105221
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        28.8031207707063
                        * (0.170655556838736 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (0.376936461235205 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0254209661646174
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            28.8031207707063
                            * (-0.292026817763947 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.168826707824519 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (-0.292026817763947 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.168826707824519 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0254209661646174
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        28.8031207707063
                        * (-0.292026817763947 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (-0.168826707824519 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.00365601748771598
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            28.8031207707063
                            * (0.466799636494498 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.403623280335378 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (0.466799636494498 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.403623280335378 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.00365601748771598
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        28.8031207707063
                        * (0.466799636494498 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (0.403623280335378 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.109257824853306
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            28.8031207707063
                            * (0.23547573937596 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.250945430575571 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (0.23547573937596 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.250945430575571 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.109257824853306
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        28.8031207707063
                        * (0.23547573937596 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (0.250945430575571 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.480286507430443
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            28.8031207707063
                            * (0.110566185432721 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.301154017333651 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (0.110566185432721 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.301154017333651 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.480286507430443
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        28.8031207707063
                        * (0.110566185432721 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (-0.301154017333651 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.105196793968471
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            28.8031207707063
                            * (-0.358055339376858 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.0107111341783262 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (-0.358055339376858 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (-0.0107111341783262 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.105196793968471
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        28.8031207707063
                        * (-0.358055339376858 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (-0.0107111341783262 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0375716901420036
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            28.8031207707063
                            * (0.0785585414434151 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.462551216897219 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            28.8031207707063
                            * (0.0785585414434151 - 0.166666666666667 * x1) ** 2
                            + 32.7180515537385
                            * (0.462551216897219 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0375716901420036
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        28.8031207707063
                        * (0.0785585414434151 - 0.166666666666667 * x1) ** 2
                        + 32.7180515537385
                        * (0.462551216897219 - 0.166666666666667 * x2) ** 2
                    )
                )
            )
            - 0.104337166479966
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
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy(),
                result.detach().cpu().numpy(),
            )
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, result))

    return result


def kriging_peaks_red050(x, results, trial, version='numpy'):
    """
    Implementation of the kriging peaks red-050function from the MINLP library.
    This function has a global minimum of -1.1566. This
    was found by SCIP

    Parameters:
        x: (x1, x2) this is a 3D problem
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
            -1.28655551917808
            * (
                (
                    0.0511758437475303
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (0.41444408088704 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.499827877577323 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.41444408088704 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.499827877577323 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0511758437475303
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (0.41444408088704 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.499827877577323 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0691984672538374
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (0.0122706395386228 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.158167788335838 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.0122706395386228 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.158167788335838 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0691984672538374
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (0.0122706395386228 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.158167788335838 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.103037403006707
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (0.328588651321815 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.422821824461503 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.328588651321815 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.422821824461503 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.103037403006707
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (0.328588651321815 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.422821824461503 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0424480339455308
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (-0.303929995021759 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.241204209262985 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.303929995021759 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.241204209262985 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0424480339455308
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (-0.303929995021759 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.241204209262985 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0632313488526626
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (0.425574140459005 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.259134777875101 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.425574140459005 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.259134777875101 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0632313488526626
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (0.425574140459005 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.259134777875101 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.132209552581735
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (0.0447064248308813 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.428502116080811 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.0447064248308813 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.428502116080811 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.132209552581735
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (0.0447064248308813 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.428502116080811 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.725429993733935
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (0.148362902134195 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.300190289364517 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.148362902134195 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.300190289364517 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.725429993733935
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (0.148362902134195 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.300190289364517 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0719930396872828
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (0.393120932447526 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.267134170090178 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.393120932447526 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.267134170090178 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0719930396872828
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (0.393120932447526 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.267134170090178 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0105495774770918
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (0.279840229982808 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.384771626047582 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.279840229982808 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.384771626047582 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0105495774770918
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (0.279840229982808 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.384771626047582 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0548977272076999
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (0.480244861711598 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.267103933501235 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.480244861711598 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.267103933501235 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0548977272076999
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (0.480244861711598 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.267103933501235 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.357003931860095
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (-0.328576906603956 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.0203517437128941 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.328576906603956 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.0203517437128941 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.357003931860095
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (-0.328576906603956 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.0203517437128941 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.00998484400329823
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (-0.455767171629523 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.146012035209209 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.455767171629523 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.146012035209209 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.00998484400329823
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (-0.455767171629523 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.146012035209209 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.162739281015443
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (0.191411127164531 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.449521423561418 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.191411127164531 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.449521423561418 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.162739281015443
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (0.191411127164531 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.449521423561418 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.185295904937941
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (-0.192660913719185 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.323718436988909 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.192660913719185 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.323718436988909 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.185295904937941
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (-0.192660913719185 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.323718436988909 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0204374171584726
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (-0.343012940406962 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.204190234776601 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.343012940406962 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.204190234776601 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0204374171584726
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (-0.343012940406962 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.204190234776601 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.268451061963553
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (0.214276281917298 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.287100183039456 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.214276281917298 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.287100183039456 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.268451061963553
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (0.214276281917298 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.287100183039456 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0744865618655468
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (-0.132597047850171 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.460440396956374 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.132597047850171 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.460440396956374 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0744865618655468
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (-0.132597047850171 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.460440396956374 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0203288837495528
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (0.0872311261877746 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.0792450014133742 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.0872311261877746 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.0792450014133742 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0203288837495528
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (0.0872311261877746 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.0792450014133742 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0252450218287266
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (-0.170511621416658 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.388756970308057 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.170511621416658 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.388756970308057 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0252450218287266
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (-0.170511621416658 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.388756970308057 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.661564139366007
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (-0.0735455379022437 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.0574103339777237 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.0735455379022437 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.0574103339777237 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.661564139366007
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (-0.0735455379022437 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.0574103339777237 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.161732335312375
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (0.118294064225034 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.492295682458604 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.118294064225034 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.492295682458604 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.161732335312375
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (0.118294064225034 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.492295682458604 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.105846110822845
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (-0.212160160433318 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.355172813565459 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.212160160433318 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.355172813565459 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.105846110822845
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (-0.212160160433318 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.355172813565459 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0017527700875928
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (0.0689500328429836 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.104419102860497 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.0689500328429836 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.104419102860497 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0017527700875928
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (0.0689500328429836 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.104419102860497 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.19505169231069
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (-0.0951308180348652 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.160090190214414 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.0951308180348652 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.160090190214414 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.19505169231069
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (-0.0951308180348652 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.160090190214414 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0258472359216103
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (0.372708180136192 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.132747129594836 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.372708180136192 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.132747129594836 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0258472359216103
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (0.372708180136192 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.132747129594836 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0575871158209796
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (0.468711755891586 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.181356877084408 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.468711755891586 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.181356877084408 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0575871158209796
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (0.468711755891586 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.181356877084408 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.00760418661448029
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (-0.0392463314673574 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.0392448315013773 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.0392463314673574 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.0392448315013773 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.00760418661448029
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (-0.0392463314673574 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.0392448315013773 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.10975082236287
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (0.22652385482879 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.316356936855873 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.22652385482879 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.316356936855873 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.10975082236287
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (0.22652385482879 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.316356936855873 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.05568255489677
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (0.457248446547711 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.0917926014491254 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.457248446547711 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.0917926014491254 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.05568255489677
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (0.457248446547711 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.0917926014491254 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.00226915572308246
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (0.294750682614849 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.415502879110356 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.294750682614849 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.415502879110356 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.00226915572308246
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (0.294750682614849 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.415502879110356 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.254616876605353
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (-0.272152627746231 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.181203760209943 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.272152627746231 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.181203760209943 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.254616876605353
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (-0.272152627746231 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.181203760209943 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0368273075616428
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (0.172167890972669 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.408892360525134 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.172167890972669 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.408892360525134 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0368273075616428
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (0.172167890972669 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.408892360525134 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.51783512021836
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (-0.110759677342194 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.355065265920299 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.110759677342194 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.355065265920299 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.51783512021836
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (-0.110759677342194 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.355065265920299 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0417508477511705
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (-0.249580175536119 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.376041484937811 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.249580175536119 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.376041484937811 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0417508477511705
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (-0.249580175536119 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.376041484937811 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0782329684838529
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (0.251338884270706 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.331768459394919 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.251338884270706 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.331768459394919 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0782329684838529
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (0.251338884270706 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.331768459394919 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.335541366644534
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (-0.0573741451334865 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.0146400960879752 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.0573741451334865 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.0146400960879752 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.335541366644534
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (-0.0573741451334865 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.0146400960879752 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.419528638513539
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (0.130060774132016 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.365973040158068 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.130060774132016 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.365973040158068 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.419528638513539
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (0.130060774132016 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.365973040158068 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0245350733443896
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (-0.298432581918246 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.445815598976878 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.298432581918246 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.445815598976878 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0245350733443896
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (-0.298432581918246 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.445815598976878 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0161746964002097
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (-0.431199720184307 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.229069908135606 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.431199720184307 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.229069908135606 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0161746964002097
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (-0.431199720184307 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.229069908135606 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    1.12186401997631
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (0.0256658827631951 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.206627423418834 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.0256658827631951 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.206627423418834 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 1.12186401997631
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (0.0256658827631951 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.206627423418834 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.349565676037355
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (0.314666152229045 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.0600057493934595 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.314666152229045 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.0600057493934595 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.349565676037355
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (0.314666152229045 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.0600057493934595 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0157021560439885
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (-0.499505162739216 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.170894347675891 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.499505162739216 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.170894347675891 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0157021560439885
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (-0.499505162739216 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.170894347675891 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.258678957923669
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (-0.153892778175042 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.100300738847656 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.153892778175042 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.100300738847656 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.258678957923669
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (-0.153892778175042 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.100300738847656 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0626591557365824
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (-0.383412829353937 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.084598585997641 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.383412829353937 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.084598585997641 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0626591557365824
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (-0.383412829353937 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.084598585997641 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.00770573000942261
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (-0.401856432968689 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.296176222487051 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.401856432968689 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.296176222487051 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.00770573000942261
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (-0.401856432968689 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.296176222487051 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.110001583263663
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (-0.015883752117149 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.468040087195413 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.015883752117149 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.468040087195413 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.110001583263663
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (-0.015883752117149 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.468040087195413 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.000311601939297733
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (-0.474180422249378 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.135156447888888 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.474180422249378 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.135156447888888 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.000311601939297733
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (-0.474180422249378 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.135156447888888 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.173777129663267
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (0.347199435240466 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.0143658105676452 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.347199435240466 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.0143658105676452 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.173777129663267
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (0.347199435240466 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.0143658105676452 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0644958976913963
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (-0.221277743362768 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.229545496211592 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.221277743362768 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.229545496211592 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0644958976913963
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (-0.221277743362768 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.229545496211592 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.125701662342689
                    * (
                        2.23606797749979
                        * np.sqrt(
                            63.938104949048
                            * (-0.370040226156181 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.0589456069978705 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.370040226156181 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.0589456069978705 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.125701662342689
                )
                * np.exp(
                    -2.23606797749979
                    * np.sqrt(
                        63.938104949048
                        * (-0.370040226156181 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.0589456069978705 - 0.166666666666667 * x2) ** 2
                    )
                )
            )
            + 0.270964000255915
        )
    elif version == 'pytorch':
        result = (
            -1.28655551917808
            * (
                (
                    0.0511758437475303
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (0.41444408088704 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.499827877577323 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.41444408088704 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.499827877577323 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0511758437475303
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (0.41444408088704 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.499827877577323 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0691984672538374
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (0.0122706395386228 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.158167788335838 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.0122706395386228 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.158167788335838 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0691984672538374
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (0.0122706395386228 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.158167788335838 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.103037403006707
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (0.328588651321815 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.422821824461503 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.328588651321815 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.422821824461503 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.103037403006707
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (0.328588651321815 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.422821824461503 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0424480339455308
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (-0.303929995021759 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.241204209262985 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.303929995021759 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.241204209262985 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0424480339455308
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (-0.303929995021759 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.241204209262985 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0632313488526626
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (0.425574140459005 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.259134777875101 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.425574140459005 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.259134777875101 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0632313488526626
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (0.425574140459005 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.259134777875101 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.132209552581735
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (0.0447064248308813 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.428502116080811 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.0447064248308813 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.428502116080811 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.132209552581735
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (0.0447064248308813 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.428502116080811 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.725429993733935
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (0.148362902134195 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.300190289364517 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.148362902134195 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.300190289364517 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.725429993733935
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (0.148362902134195 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.300190289364517 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0719930396872828
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (0.393120932447526 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.267134170090178 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.393120932447526 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.267134170090178 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0719930396872828
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (0.393120932447526 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.267134170090178 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0105495774770918
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (0.279840229982808 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.384771626047582 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.279840229982808 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.384771626047582 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0105495774770918
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (0.279840229982808 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.384771626047582 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0548977272076999
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (0.480244861711598 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.267103933501235 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.480244861711598 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.267103933501235 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0548977272076999
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (0.480244861711598 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.267103933501235 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.357003931860095
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (-0.328576906603956 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.0203517437128941 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.328576906603956 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.0203517437128941 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.357003931860095
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (-0.328576906603956 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.0203517437128941 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.00998484400329823
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (-0.455767171629523 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.146012035209209 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.455767171629523 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.146012035209209 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.00998484400329823
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (-0.455767171629523 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.146012035209209 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.162739281015443
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (0.191411127164531 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.449521423561418 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.191411127164531 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.449521423561418 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.162739281015443
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (0.191411127164531 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.449521423561418 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.185295904937941
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (-0.192660913719185 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.323718436988909 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.192660913719185 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.323718436988909 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.185295904937941
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (-0.192660913719185 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.323718436988909 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0204374171584726
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (-0.343012940406962 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.204190234776601 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.343012940406962 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.204190234776601 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0204374171584726
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (-0.343012940406962 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.204190234776601 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.268451061963553
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (0.214276281917298 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.287100183039456 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.214276281917298 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.287100183039456 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.268451061963553
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (0.214276281917298 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.287100183039456 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0744865618655468
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (-0.132597047850171 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.460440396956374 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.132597047850171 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.460440396956374 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0744865618655468
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (-0.132597047850171 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.460440396956374 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0203288837495528
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (0.0872311261877746 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.0792450014133742 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.0872311261877746 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.0792450014133742 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0203288837495528
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (0.0872311261877746 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.0792450014133742 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0252450218287266
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (-0.170511621416658 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.388756970308057 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.170511621416658 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.388756970308057 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0252450218287266
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (-0.170511621416658 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.388756970308057 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.661564139366007
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (-0.0735455379022437 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.0574103339777237 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.0735455379022437 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.0574103339777237 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.661564139366007
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (-0.0735455379022437 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.0574103339777237 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.161732335312375
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (0.118294064225034 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.492295682458604 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.118294064225034 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.492295682458604 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.161732335312375
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (0.118294064225034 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.492295682458604 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.105846110822845
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (-0.212160160433318 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.355172813565459 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.212160160433318 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.355172813565459 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.105846110822845
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (-0.212160160433318 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.355172813565459 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0017527700875928
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (0.0689500328429836 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.104419102860497 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.0689500328429836 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.104419102860497 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0017527700875928
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (0.0689500328429836 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.104419102860497 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.19505169231069
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (-0.0951308180348652 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.160090190214414 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.0951308180348652 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.160090190214414 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.19505169231069
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (-0.0951308180348652 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.160090190214414 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0258472359216103
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (0.372708180136192 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.132747129594836 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.372708180136192 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.132747129594836 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0258472359216103
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (0.372708180136192 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.132747129594836 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0575871158209796
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (0.468711755891586 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.181356877084408 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.468711755891586 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.181356877084408 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0575871158209796
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (0.468711755891586 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.181356877084408 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.00760418661448029
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (-0.0392463314673574 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.0392448315013773 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.0392463314673574 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.0392448315013773 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.00760418661448029
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (-0.0392463314673574 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.0392448315013773 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.10975082236287
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (0.22652385482879 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.316356936855873 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.22652385482879 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.316356936855873 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.10975082236287
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (0.22652385482879 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.316356936855873 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.05568255489677
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (0.457248446547711 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.0917926014491254 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.457248446547711 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.0917926014491254 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.05568255489677
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (0.457248446547711 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.0917926014491254 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.00226915572308246
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (0.294750682614849 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.415502879110356 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.294750682614849 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.415502879110356 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.00226915572308246
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (0.294750682614849 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.415502879110356 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.254616876605353
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (-0.272152627746231 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.181203760209943 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.272152627746231 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.181203760209943 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.254616876605353
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (-0.272152627746231 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.181203760209943 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0368273075616428
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (0.172167890972669 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.408892360525134 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.172167890972669 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.408892360525134 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0368273075616428
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (0.172167890972669 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.408892360525134 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.51783512021836
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (-0.110759677342194 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.355065265920299 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.110759677342194 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.355065265920299 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.51783512021836
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (-0.110759677342194 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.355065265920299 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0417508477511705
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (-0.249580175536119 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.376041484937811 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.249580175536119 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.376041484937811 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0417508477511705
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (-0.249580175536119 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.376041484937811 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0782329684838529
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (0.251338884270706 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.331768459394919 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.251338884270706 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.331768459394919 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0782329684838529
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (0.251338884270706 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.331768459394919 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.335541366644534
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (-0.0573741451334865 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.0146400960879752 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.0573741451334865 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.0146400960879752 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.335541366644534
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (-0.0573741451334865 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.0146400960879752 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.419528638513539
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (0.130060774132016 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.365973040158068 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.130060774132016 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.365973040158068 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.419528638513539
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (0.130060774132016 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.365973040158068 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0245350733443896
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (-0.298432581918246 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.445815598976878 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.298432581918246 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.445815598976878 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0245350733443896
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (-0.298432581918246 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.445815598976878 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0161746964002097
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (-0.431199720184307 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.229069908135606 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.431199720184307 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.229069908135606 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0161746964002097
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (-0.431199720184307 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.229069908135606 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    1.12186401997631
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (0.0256658827631951 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.206627423418834 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.0256658827631951 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.206627423418834 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 1.12186401997631
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (0.0256658827631951 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.206627423418834 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.349565676037355
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (0.314666152229045 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.0600057493934595 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.314666152229045 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.0600057493934595 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.349565676037355
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (0.314666152229045 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.0600057493934595 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0157021560439885
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (-0.499505162739216 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.170894347675891 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.499505162739216 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.170894347675891 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0157021560439885
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (-0.499505162739216 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.170894347675891 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.258678957923669
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (-0.153892778175042 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.100300738847656 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.153892778175042 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.100300738847656 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.258678957923669
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (-0.153892778175042 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.100300738847656 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.0626591557365824
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (-0.383412829353937 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.084598585997641 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.383412829353937 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.084598585997641 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0626591557365824
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (-0.383412829353937 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.084598585997641 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.00770573000942261
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (-0.401856432968689 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.296176222487051 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.401856432968689 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.296176222487051 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.00770573000942261
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (-0.401856432968689 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.296176222487051 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.110001583263663
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (-0.015883752117149 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.468040087195413 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.015883752117149 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.468040087195413 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.110001583263663
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (-0.015883752117149 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.468040087195413 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.000311601939297733
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (-0.474180422249378 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.135156447888888 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.474180422249378 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.135156447888888 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.000311601939297733
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (-0.474180422249378 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.135156447888888 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.173777129663267
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (0.347199435240466 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.0143658105676452 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (0.347199435240466 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.0143658105676452 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.173777129663267
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (0.347199435240466 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.0143658105676452 - 0.166666666666667 * x2) ** 2
                    )
                )
                - (
                    0.0644958976913963
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (-0.221277743362768 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.229545496211592 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.221277743362768 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (-0.229545496211592 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.0644958976913963
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (-0.221277743362768 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (-0.229545496211592 - 0.166666666666667 * x2) ** 2
                    )
                )
                + (
                    0.125701662342689
                    * (
                        2.23606797749979
                        * torch.sqrt(
                            63.938104949048
                            * (-0.370040226156181 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.0589456069978705 - 0.166666666666667 * x2) ** 2
                        )
                        + 1.66666666666667
                        * (
                            63.938104949048
                            * (-0.370040226156181 - 0.166666666666667 * x1) ** 2
                            + 11.9516380997938
                            * (0.0589456069978705 - 0.166666666666667 * x2) ** 2
                        )
                    )
                    + 0.125701662342689
                )
                * torch.exp(
                    -2.23606797749979
                    * torch.sqrt(
                        63.938104949048
                        * (-0.370040226156181 - 0.166666666666667 * x1) ** 2
                        + 11.9516380997938
                        * (0.0589456069978705 - 0.166666666666667 * x2) ** 2
                    )
                )
            )
            + 0.270964000255915
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
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy(),
                result.detach().cpu().numpy(),
            )
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, result))

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
    'bounds': [(-32.768, 32.768)],  # Will use a single level bound and then expand
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

# Multi-Dimensional Problems #
ackley_30d_config = {
    'objective': ndackley,
    'bounds': [(-32.768, 32.768)],  # Will use a single level bound and then expand
    'max_iterations': 1000,
    'global_minimum': 0.0,
    'dimensions': 30,
}

# ex8_6_2 from MINLP
ex8_6_2_config = {
    'objective': ex8_6_2,
    'bounds': [
        (-1e-6, 1e-6),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-1e-6, 1e-6),
        (-1e-6, 1e-6),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-1e-6, 1e-6),
        (-1e-6, 1e-6),
        (-1e-6, 1e-6),
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

# least from MINLP
least_config = {
    'objective': least,
    'bounds': [
        (None, None),
        (None, None),
        (-5.0, 5.0),
    ],
    'max_iterations': 1000,
    'global_minimum': 0.0,
    'dimensions': 3,
}

# ex4_1_5 from MINLP
ex4_1_5_config = {
    'objective': ex4_1_5,
    'bounds': [
        (-5, None),
        (None, 5),
    ],
    'max_iterations': 1000,
    'global_minimum': 0.0,
    'dimensions': 2,
}

ex8_1_1_config = {
    'objective': ex8_1_1,
    'bounds': [
        (-1, 2),
        (-1, 1),
    ],
    'max_iterations': 1000,
    'global_minimum': -2.02180678,
    'dimensions': 2,
}

ex8_1_3_config = {
    'objective': ex8_1_3,
    'bounds': [
        (None, None),
        (None, None),
    ],
    'max_iterations': 1000,
    'global_minimum': 3.0,
    'dimensions': 2,
}

ex8_1_4_config = {
    'objective': ex8_1_4,
    'bounds': [
        (None, None),
        (None, None),
    ],
    'max_iterations': 1000,
    'global_minimum': 0.0,
    'dimensions': 2,
}

ex8_1_6_config = {
    'objective': ex8_1_6,
    'bounds': [
        (None, None),
        (None, None),
    ],
    'max_iterations': 1000,
    'global_minimum': -10.0860,
    'dimensions': 2,
}

kriging_peaks_red010_config = {
    'objective': kriging_peaks_red010,
    'bounds': [
        (-3, 3),
        (-3, 3),
    ],
    'max_iterations': 1000,
    'global_minimum': 0.2911,
    'dimensions': 2,
}

kriging_peaks_red020_config = {
    'objective': kriging_peaks_red020,
    'bounds': [
        (-3, 3),
        (-3, 3),
    ],
    'max_iterations': 1000,
    'global_minimum': 0.3724,
    'dimensions': 2,
}

kriging_peaks_red030_config = {
    'objective': kriging_peaks_red030,
    'bounds': [
        (-3, 3),
        (-3, 3),
    ],
    'max_iterations': 1000,
    'global_minimum': -1.5886,
    'dimensions': 2,
}

kriging_peaks_red050_config = {
    'objective': kriging_peaks_red050,
    'bounds': [
        (-3, 3),
        (-3, 3),
    ],
    'max_iterations': 1000,
    'global_minimum': -1.1566,
    'dimensions': 2,
}

kriging_peaks_red100_config = {
    'objective': kriging_peaks_red100,
    'bounds': [
        (-3, 3),
        (-3, 3),
    ],
    'max_iterations': 1000,
    'global_minimum': -2.6375,
    'dimensions': 2,
}

PROBLEMS_BY_NAME = {
    'ackley': ackley_config,
    'ackley_3d': ackley_3d_config,
    'ackley_5d': ackley_5d_config,
    'ackley_30d': ackley_30d_config,
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
    'least': least_config,
    'ex4_1_5': ex4_1_5_config,
    'ex8_1_1': ex8_1_1_config,
    'ex8_1_3': ex8_1_3_config,
    'ex8_1_5': ex4_1_5_config,
    'ex8_1_4': ex8_1_4_config,
    'ex8_1_6': ex8_1_6_config,
    'kriging_peaks_red010': kriging_peaks_red010_config,
    'kriging_peaks_red020': kriging_peaks_red020_config,
    'kriging_peaks_red030': kriging_peaks_red030_config,
    'kriging_peaks_red050': kriging_peaks_red050_config,
    'kriging_peaks_red100': kriging_peaks_red100_config,
}
