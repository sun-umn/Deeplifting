# third party
import numpy as np
import torch
from scipy.special import gamma

# first party
from deeplifting.kriging_peaks.kriging_peaks_red import (
    kriging_peaks_red010,
    kriging_peaks_red020,
    kriging_peaks_red030,
    kriging_peaks_red050,
    kriging_peaks_red100,
    kriging_peaks_red200,
    kriging_peaks_red500,
)


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


def mathopt6(x, results, trial, version='numpy'):
    """
    Implementation of the mathopt6 function from the MINLP library.
    This function has a global minimum of -3.306868. This
    was found by COUENNE

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
            np.exp(np.sin(50 * x1))
            + np.sin(60 * np.exp(x2))
            + np.sin(70 * np.sin(x1))
            + np.sin(np.sin(80 * x2))
            - np.sin(10 * x1 + 10 * x2)
            + 0.25 * (x1**2 + x2**2)
        )
    elif version == 'pytorch':
        result = (
            torch.exp(torch.sin(50 * x1))
            + torch.sin(60 * torch.exp(x2))
            + torch.sin(70 * torch.sin(x1))
            + torch.sin(torch.sin(80 * x2))
            - torch.sin(10 * x1 + 10 * x2)
            + 0.25 * (x1**2 + x2**2)
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


def quantum(x, results, trial, version='numpy'):
    """
    Implementation of the quantum function from the MINLP library.
    This function has a global minimum of 0.8049. This
    was found by CONOPT

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
            0.5 * np.square(x2) * gamma(2 - 0.5 / x2) / gamma(0.5 / x2) * x1 ** (1 / x2)
            + 0.5 * gamma(1.5 / x2) / gamma(0.5 / x2) * x1 ** (-1 / x2)
            + gamma(2.5 / x2) / gamma(0.5 / x2) * x1 ** (-2 / x2)
        )
    elif version == 'pytorch':
        result = (
            0.5
            * torch.square(x2)
            * torch.exp(torch.lgamma(2 - 0.5 / x2))
            / torch.exp(torch.lgamma(0.5 / x2))
            * x1 ** (1 / x2)
            + 0.5
            * torch.exp(torch.lgamma(1.5 / x2))
            / torch.exp(torch.lgamma(0.5 / x2))
            * x1 ** (-1 / x2)
            + torch.exp(torch.lgamma(2.5 / x2))
            / torch.exp(torch.lgamma(0.5 / x2))
            * x1 ** (-2 / x2)
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


def rosenbrock(x, results, trial, version='numpy'):
    """
    Implementation of the rosenbrock function from the MINLP library.
    This function has a global minimum of 0.0. This
    was found by CONOPT

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
        result = 100 * (-(x1**2) + x2) ** 2 + (1 - x1) ** 2
    elif version == 'pytorch':
        result = 100 * (-(x1**2) + x2) ** 2 + (1 - x1) ** 2
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

ex8_1_5_config = {
    'objective': ex8_1_5,
    'bounds': [
        (None, None),
        (None, None),
    ],
    'max_iterations': 1000,
    'global_minimum': -1.0316,
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

kriging_peaks_red200_config = {
    'objective': kriging_peaks_red200,
    'bounds': [
        (-3, 3),
        (-3, 3),
    ],
    'max_iterations': 1000,
    'global_minimum': -3.8902,
    'dimensions': 2,
}

kriging_peaks_red500_config = {
    'objective': kriging_peaks_red500,
    'bounds': [
        (-3, 3),
        (-3, 3),
    ],
    'max_iterations': 1000,
    'global_minimum': -4.9280,
    'dimensions': 2,
}

mathopt6_config = {
    'objective': mathopt6,
    'bounds': [
        (-3, 3),
        (-3, 3),
    ],
    'max_iterations': 1000,
    'global_minimum': -3.3069,
    'dimensions': 2,
}

quantum_config = {
    'objective': quantum,
    'bounds': [
        (1, 10),
        (1, 10),
    ],
    'max_iterations': 1000,
    'global_minimum': 0.8049,
    'dimensions': 2,
}

rosenbrock_config = {
    'objective': rosenbrock,
    'bounds': [
        (-10, 5),
        (-10, 10),
    ],
    'max_iterations': 1000,
    'global_minimum': 0.8049,
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
    'ex8_1_5': ex8_1_5_config,
    'ex8_1_4': ex8_1_4_config,
    'ex8_1_6': ex8_1_6_config,
    'kriging_peaks_red010': kriging_peaks_red010_config,
    'kriging_peaks_red020': kriging_peaks_red020_config,
    'kriging_peaks_red030': kriging_peaks_red030_config,
    'kriging_peaks_red050': kriging_peaks_red050_config,
    'kriging_peaks_red100': kriging_peaks_red100_config,
    'kriging_peaks_red200': kriging_peaks_red200_config,
    'kriging_peaks_red500': kriging_peaks_red500_config,
    'mathopt6': mathopt6_config,
    'quantum': quantum_config,
    'rosenbrock': rosenbrock_config,
}
