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


def ndgriewank(x, results, trial, version='numpy'):
    """
    Implementation of the n-dimensional Griewank function

    Args:
    x: A d-dimensional array or tensor
    version: A string, either 'numpy' or 'pytorch'

    Returns:
    result: Value of the griewank function
    """
    d = len(x)
    x = x.flatten()

    if version == 'numpy':
        sqrt_i = np.sqrt(np.arange(1, d + 1)).flatten()
        result = np.sum(np.square(x) / 4000) - np.prod(np.cos(x / sqrt_i)) + 1
    elif version == 'pytorch':
        sqrt_i = torch.sqrt(torch.arange(1, d + 1)).flatten()
        result = (
            torch.sum(torch.square(x) / 4000) - torch.prod(torch.cos(x / sqrt_i)) + 1
        )
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

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


def langermann(x, results, trial, version='numpy'):
    """
    Implementation of the 2D Langermann function.

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
        result = (
            np.exp((-1 / np.pi) * (np.square(x1 - 3) + np.square(x2 - 5)))
            * np.cos(np.pi * (np.square(x1 - 3) + np.square(x2 - 5)))
            + 2
            * np.exp((-1 / np.pi) * (np.square(x1 - 5) + np.square(x2 - 2)))
            * np.cos(np.pi * (np.square(x1 - 5) + np.square(x2 - 2)))
            + 5
            * np.exp((-1 / np.pi) * (np.square(x1 - 2) + np.square(x2 - 1)))
            * np.cos(np.pi * (np.square(x1 - 2) + np.square(x2 - 1)))
            + 2
            * np.exp((-1 / np.pi) * (np.square(x1 - 1) + np.square(x2 - 4)))
            * np.cos(np.pi * (np.square(x1 - 1) + np.square(x2 - 4)))
            + 3
            * np.exp((-1 / np.pi) * (np.square(x1 - 7) + np.square(x2 - 9)))
            * np.cos(np.pi * (np.square(x1 - 7) + np.square(x2 - 9)))
        )
    elif version == 'pytorch':
        result = (
            torch.exp((-1 / np.pi) * (torch.square(x1 - 3) + torch.square(x2 - 5)))
            * torch.cos(np.pi * (torch.square(x1 - 3) + torch.square(x2 - 5)))
            + 2
            * torch.exp((-1 / np.pi) * (torch.square(x1 - 5) + torch.square(x2 - 2)))
            * torch.cos(np.pi * (torch.square(x1 - 5) + torch.square(x2 - 2)))
            + 5
            * torch.exp((-1 / np.pi) * (torch.square(x1 - 2) + torch.square(x2 - 1)))
            * torch.cos(np.pi * (torch.square(x1 - 2) + torch.square(x2 - 1)))
            + 2
            * torch.exp((-1 / np.pi) * (torch.square(x1 - 1) + torch.square(x2 - 4)))
            * torch.cos(np.pi * (torch.square(x1 - 1) + torch.square(x2 - 4)))
            + 3
            * torch.exp((-1 / np.pi) * (torch.square(x1 - 7) + torch.square(x2 - 9)))
            * torch.cos(np.pi * (torch.square(x1 - 7) + torch.square(x2 - 9)))
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


def ndlevy(x, results, trial, version='numpy'):
    """
    Implemention of the n-dimensional levy function

    Args:
    x: A d-dimensional array or tensor
    version: A string, either 'numpy' or 'pytorch'

    Returns:
    result: Value of the Levy function
    """
    x = x.flatten()
    w = ((x - 1) / 2) + 1
    w1 = w[0]
    wd = w[-1]
    w = w[0:-1]
    if version == 'numpy':
        result = (
            np.square(np.sin(np.pi * w1))
            + np.sum(np.square(w - 1) * (1 + 10 * np.square(np.sin(np.pi * w + 1))))
            + np.square(wd - 1) * (1 + np.square(np.sin(2 * np.pi * wd)))
        )
    elif version == 'pytorch':
        result = (
            torch.square(torch.sin(np.pi * w1))
            + torch.sum(
                torch.square(w - 1) * (1 + 10 * torch.square(torch.sin(np.pi * w + 1)))
            )
            + torch.square(wd - 1) * (1 + torch.square(torch.sin(2 * np.pi * wd)))
        )
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

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


def ndrastrigin(x, results, trial, version='numpy'):
    """
    Implemention of the n-dimensional levy function

    Args:
    x: A d-dimensional array or tensor
    version: A string, either 'numpy' or 'pytorch'

    Returns:
    result: Value of the Rastrigin function
    """
    x = x.flatten()
    d = len(x)
    if version == 'numpy':
        result = 10 * d + np.sum(np.square(x) - 10 * np.cos(2 * np.pi * x))
    elif version == 'pytorch':
        result = 10 * d + torch.sum(torch.square(x) - 10 * torch.cos(2 * np.pi * x))
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

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


def ndschwefel(x, results, trial, version='numpy'):
    """
    Implemention of the n-dimensional levy function

    Args:
    x: A d-dimensional array or tensor
    version: A string, either 'numpy' or 'pytorch'

    Returns:
    result: Value of the Rastrigin function
    """
    x = x.flatten()
    d = len(x)
    if version == 'numpy':
        result = 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    elif version == 'pytorch':
        result == 418.9829 * d - torch.sum(x * torch.sin(torch.sqrt(torch.abs(x))))
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

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


def damavandi(x, results, trial, version='numpy'):
    """
    Implementation of the Damavandi problem from the infinity77 list.
    This is a 3-dimensional function with a global minimum of 0.0 at (2,2)

    Parameters:
        x: (x1, x2) this is a 3D problem
    version : str
        The version to use for the function's computation.
        Options are 'numpy' and 'pytorch'.

    Returns:
    result : np.ndarray or torch.Tensor
        The computed Damavandi function values
        corresponding to the inputs (x1, x2).

    Raises:
    ValueError
        If the version is not 'numpy' or 'pytorch'.
    """
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = (
            1
            - np.abs(
                (np.sin(np.pi * (x1 - 2)) * np.sin(np.pi * (x2 - 2)))
                / ((np.pi**2) * (x1 - 2) * (x2 - 2))
            )
            ** 5
        ) * (2 + (x1 - 7) ** 2 + 2 * (x2 - 7) ** 2)
    elif version == 'pytorch':
        result = (
            1
            - torch.abs(
                (torch.sin(np.pi * (x1 - 2)) * torch.sin(np.pi * (x2 - 2)))
                / ((np.pi**2) * (x1 - 2) * (x2 - 2))
            )
            ** 5
        ) * (2 + (x1 - 7) ** 2 + 2 * (x2 - 7) ** 2)
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


def cross_leg_table(x, results, trial, version='numpy'):
    """
    Implementation of the CrossLegTable problem from the infinity77 list.
    This is a 3-dimensional function with a global minimum of -1.0 at (0,0)

    Parameters:
        x: (x1, x2) this is a 3D problem
    version : str
        The version to use for the function's computation.
        Options are 'numpy' and 'pytorch'.

    Returns:
    result : np.ndarray or torch.Tensor
        The computed Damavandi function values
        corresponding to the inputs (x1, x2).

    Raises:
    ValueError
        If the version is not 'numpy' or 'pytorch'.
    """
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = -(
            1
            / (
                (
                    np.abs(
                        np.exp(np.abs(100 - (((x1**2 + x2**2) ** 0.5) / (np.pi))))
                        * np.sin(x1)
                        * np.sin(x2)
                    )
                    + 1
                )
                ** 0.1
            )
        )
    elif version == 'pytorch':
        result = -(
            1
            / (
                (
                    torch.abs(
                        torch.exp(
                            torch.abs(100 - (((x1**2 + x2**2) ** 0.5) / (np.pi)))
                        )
                        * torch.sin(x1)
                        * torch.sin(x2)
                    )
                    + 1
                )
                ** 0.1
            )
        )
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


def sine_envelope(x, results, trial, version='numpy'):
    """
    Implementation of the SineEnvelope problem from the infinity77 list.
    This is a 3-dimensional function with a global minimum of -0.72984 at (0,0)

    Parameters:
        x: (x1, x2) this is a 3D problem
    version : str
        The version to use for the function's computation.
        Options are 'numpy' and 'pytorch'.

    Returns:
    result : np.ndarray or torch.Tensor
        The computed Damavandi function values
        corresponding to the inputs (x1, x2).

    Raises:
    ValueError
        If the version is not 'numpy' or 'pytorch'.
    """
    x1, x2 = x.flatten()
    if version == 'numpy':
        r = np.sqrt(x2**2 + x1**2)
        numerator = np.sin(r) ** 2 - 0.5
        denominator = (0.001 * r**2 + 1.0) ** 2
        result = -(numerator / denominator + 0.5)
    elif version == 'pytorch':
        r = torch.sqrt(x2**2 + x1**2)
        numerator = torch.sin(r) ** 2 - 0.5
        denominator = (0.001 * r**2 + 1.0) ** 2
        result = -(numerator / denominator + 0.5)
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


# Problems from the literature survey by Jamil et al.
# Ackley2
def ackley2(x, results, trial, version='numpy'):
    """
    Implementation of the Ackley2 function.
    This is a 3-dimensional function with a global minimum of -200 at (0,0)

    Parameters:
        x: (x1, x2) this is a 3D problem
    version : str
        The version to use for the function's computation.
        Options are 'numpy' and 'pytorch'.

    Returns:
    result : np.ndarray or torch.Tensor
        The computed Damavandi function values
        corresponding to the inputs (x1, x2).

    Raises:
    ValueError
        If the version is not 'numpy' or 'pytorch'.
    """
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = -200 * np.exp(-0.02 * np.sqrt(x1**2 + x2**2))
    elif version == 'pytorch':
        result = -200 * torch.exp(-0.02 * torch.sqrt(x1**2 + x2**2))
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


# Ackley3
def ackley3(x, results, trial, version='numpy'):
    """
    Implementation of the Ackley3 function.
    This is a 3-dimensional function with a global minimum of -219.1418 at (0,~-0.4)

    Parameters:
        x: (x1, x2) this is a 3D problem
    version : str
        The version to use for the function's computation.
        Options are 'numpy' and 'pytorch'.

    Returns:
    result : np.ndarray or torch.Tensor
        The computed Damavandi function values
        corresponding to the inputs (x1, x2).

    Raises:
    ValueError
        If the version is not 'numpy' or 'pytorch'.
    """
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = 200 * np.exp(-0.02 * np.sqrt(x1**2 + x2**2)) + 5 * np.exp(
            np.cos(3 * x1) + np.sin(3 * x2)
        )
    elif version == 'pytorch':
        result = 200 * torch.exp(-0.02 * torch.sqrt(x1**2 + x2**2)) + 5 * torch.exp(
            torch.cos(3 * x1) + torch.sin(3 * x2)
        )
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


# nd Ackley4
def ndackley4(x, results, trial, version='numpy'):
    """
    Compute the Ackley4 function.

    Args:
    x: A d-dimensional array or tensor
    version: A string, either 'numpy' or 'pytorch'

    Returns:
    result: Value of the Ackley function
    """
    x = x.flatten()
    shifted_x = x.flatten()[1:]
    x = x.flatten()[:-1]
    if version == 'numpy':
        result = np.sum(
            np.exp(-0.2) * np.sqrt(np.square(x) + np.square(shifted_x))
            + 3 * (np.cos(2 * x) + np.sin(2 * shifted_x))
        )
    elif version == 'pytorch':
        result = np.sum(
            torch.exp(-0.2) * torch.sqrt(torch.square(x) + torch.square(shifted_x))
            + 3 * (torch.cos(2 * x) + torch.sin(2 * shifted_x))
        )
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

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


# Adjiman
def adjiman(x, results, trial, version='numpy'):
    """
    Implementation of the Adjiman function.
    This is a 3-dimensional function with a global minimum of -2.02181 at (2,0.10578)

    Parameters:
        x: (x1, x2) this is a 3D problem
    version : str
        The version to use for the function's computation.
        Options are 'numpy' and 'pytorch'.

    Returns:
    result : np.ndarray or torch.Tensor
        The computed Damavandi function values
        corresponding to the inputs (x1, x2).

    Raises:
    ValueError
        If the version is not 'numpy' or 'pytorch'.
    """
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = np.cos(x1) * np.sin(x2) - (x1 / (x2**2 + 1))
    elif version == 'pytorch':
        result = torch.cos(x1) * torch.sin(x2) - (x1 / (x2**2 + 1))
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


# Alpine1 in 2d
def alpine1(x, results, trial, version='numpy'):
    """
    Implementation of the Alpine1 function.
    This is a 3-dimensional function with a global minimum of 0 at (0,0)

    Parameters:
        x: (x1, x2) this is a 3D problem
    version : str
        The version to use for the function's computation.
        Options are 'numpy' and 'pytorch'.

    Returns:
    result : np.ndarray or torch.Tensor
        The computed Damavandi function values
        corresponding to the inputs (x1, x2).

    Raises:
    ValueError
        If the version is not 'numpy' or 'pytorch'.
    """
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = np.abs(x1 * np.sin(x1) + 0.1 * x1) + np.abs(x2 * np.sin(x2) + 0.1 * x1)
    elif version == 'pytorch':
        result = torch.abs(x1 * torch.sin(x1) + 0.1 * x1) + torch.abs(
            x2 * torch.sin(x2) + 0.1 * x2
        )
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


# nd Alpine1
def ndalpine1(x, results, trial, version='numpy'):
    """
    Compute the Alpine1 function.

    Args:
    x: A d-dimensional array or tensor
    version: A string, either 'numpy' or 'pytorch'

    Returns:
    result: Value of the Ackley function
    """
    x = x.flatten()
    if version == 'numpy':
        result = np.sum(np.abs(x * np.sin(x) + 0.1 * x))
    elif version == 'pytorch':
        result = torch.sum(torch.abs(x * torch.sin(x) + 0.1 * x))
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

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


# Alpine2 in 2d
def alpine2(x, results, trial, version='numpy'):
    """
    Implementation of the Alpine2 function.
    This is a 3-dimensional function with a global minimum of 2.8^2 at (7.917,7.917)

    Parameters:
        x: (x1, x2) this is a 3D problem
    version : str
        The version to use for the function's computation.
        Options are 'numpy' and 'pytorch'.

    Returns:
    result : np.ndarray or torch.Tensor
        The computed Damavandi function values
        corresponding to the inputs (x1, x2).

    Raises:
    ValueError
        If the version is not 'numpy' or 'pytorch'.
    """
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = (np.sqrt(x1) * np.sin(x1)) * (np.sqrt(x2) * np.sin(x2))
    elif version == 'pytorch':
        result = (torch.sqrt(x1) * torch.sin(x1)) * (torch.sqrt(x2) * torch.sin(x2))
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


# nd Alpine2
def ndalpine2(x, results, trial, version='numpy'):
    """
    Compute the Alpine2 function.

    Args:
    x: A d-dimensional array or tensor
    version: A string, either 'numpy' or 'pytorch'

    Returns:
    result: Value of the Ackley function
    """
    x = x.flatten()
    if version == 'numpy':
        result = np.prod(np.sqrt(x) * np.sin(x))
    elif version == 'pytorch':
        result = torch.prod(torch.sqrt(x) * torch.sin(x))
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

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


# Brad in 3d
def brad(x, results, trial, version='numpy'):
    x1, x2, x3 = x.flatten()
    if version == 'numpy':
        u = np.arange(1, 16)
        v = 16 - u
        w = np.minimum(u, v)
        y = np.array(
            [
                0.14,
                0.18,
                0.22,
                0.25,
                0.29,
                0.32,
                0.35,
                0.39,
                0.37,
                0.58,
                0.73,
                0.96,
                1.34,
                2.10,
                4.39,
            ]
        )
        result = np.sum(np.square((y - x1 - u) / (v * x2 + w * x3)))
    elif version == 'pytorch':
        u = torch.arange(1, 16)
        v = 16 - u
        w = torch.minimum(u, v)
        y = torch.array(
            [
                0.14,
                0.18,
                0.22,
                0.25,
                0.29,
                0.32,
                0.35,
                0.39,
                0.37,
                0.58,
                0.73,
                0.96,
                1.34,
                2.10,
                4.39,
            ]
        )
        result = torch.sum(torch.square((y - x1 - u) / (v * x2 + w * x3)))
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


# Bartels Conn in 2d
def bartels_conn(x, results, trial, version='numpy'):
    """
    Implementation of the Bartels Conn function.
    This is a 3-dimensional function with a global minimum of 1 at (0,0)

    Parameters:
        x: (x1, x2) this is a 3D problem
    version : str
        The version to use for the function's computation.
        Options are 'numpy' and 'pytorch'.

    Returns:
    result : np.ndarray or torch.Tensor
        The computed Damavandi function values
        corresponding to the inputs (x1, x2).

    Raises:
    ValueError
        If the version is not 'numpy' or 'pytorch'.
    """
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = (
            np.abs(np.square(x1) + np.square(x2) + x1 * x2)
            + np.abs(np.sin(x1))
            + np.abs(np.cos(x2))
        )
    elif version == 'pytorch':
        result = (
            torch.abs(torch.square(x1) + torch.square(x2) + x1 * x2)
            + torch.abs(torch.sin(x1))
            + torch.abs(torch.cos(x2))
        )
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


# Beale in 2d
def beale(x, results, trial, version='numpy'):
    """
    Implementation of the Beale function.
    This is a 3-dimensional function with a global minimum of 0 at (3,0.5)

    Parameters:
        x: (x1, x2) this is a 3D problem
    version : str
        The version to use for the function's computation.
        Options are 'numpy' and 'pytorch'.

    Returns:
    result : np.ndarray or torch.Tensor
        The computed Damavandi function values
        corresponding to the inputs (x1, x2).

    Raises:
    ValueError
        If the version is not 'numpy' or 'pytorch'.
    """
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = (
            np.square(1.5 - x1 + x1 * x2)
            + np.square((2.25 - x1 + x1 * np.square(x2)))
            + np.square(2.625 - x1 + x1 * np.power(x2, 3))
        )
    elif version == 'pytorch':
        result = (
            torch.square(1.5 - x1 + x1 * x2)
            + torch.square((2.25 - x1 + x1 * torch.square(x2)))
            + torch.square(2.625 - x1 + x1 * torch.pow(x2, 3))
        )
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


# Biggs EXP2 in 2d
def biggs_exp2(x, results, trial, version='numpy'):
    """
    Implementation of the Biggs EXP2 function.
    This is a 3-dimensional function with a global minimum of 0 at (1,10)

    Parameters:
        x: (x1, x2) this is a 3D problem
    version : str
        The version to use for the function's computation.
        Options are 'numpy' and 'pytorch'.

    Returns:
    result : np.ndarray or torch.Tensor
        The computed Damavandi function values
        corresponding to the inputs (x1, x2).

    Raises:
    ValueError
        If the version is not 'numpy' or 'pytorch'.
    """
    x1, x2 = x.flatten()
    if version == 'numpy':
        t = np.arange(1, 11) * 0.1
        y = np.exp(-t) - 5 * np.exp(10 * t)
        result = np.sum(np.square(np.exp(-t * x1) - 5 * np.exp(-t * x2) - y))
    elif version == 'pytorch':
        t = torch.arange(1, 11) * 0.1
        y = torch.exp(-t) - 5 * torch.exp(10 * t)
        result = torch.sum(
            torch.square(torch.exp(-t * x1) - 5 * torch.exp(-t * x2) - y)
        )
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


# Biggs EXP3 in 3d
def biggs_exp3(x, results, trial, version='numpy'):
    """
    Implementation of the Biggs EXP3 function.
    This is a 4-dimensional function with a global minimum of 0 at (1,10,5)

    Parameters:
        x: (x1, x2, x3) this is a 4D problem
    version : str
        The version to use for the function's computation.
        Options are 'numpy' and 'pytorch'.

    Returns:
    result : np.ndarray or torch.Tensor
        The computed Damavandi function values
        corresponding to the inputs (x1, x2, x3).

    Raises:
    ValueError
        If the version is not 'numpy' or 'pytorch'.
    """
    x1, x2, x3 = x.flatten()
    if version == 'numpy':
        t = np.arange(1, 11) * 0.1
        y = np.exp(-t) - 5 * np.exp(10 * t)
        result = np.sum(np.square(np.exp(-t * x1) - x3 * np.exp(-t * x2) - y))
    elif version == 'pytorch':
        t = torch.arange(1, 11) * 0.1
        y = torch.exp(-t) - 5 * torch.exp(10 * t)
        result = torch.sum(
            torch.square(torch.exp(-t * x1) - x3 * torch.exp(-t * x2) - y)
        )
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
                x3.detach().cpu().numpy(),
                result.detach().cpu().numpy(),
            )
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, x3, result))

    return result


# Biggs EXP4 in 4d
def biggs_exp4(x, results, trial, version='numpy'):
    """
    Implementation of the Biggs EXP4 function.
    This is a 5-dimensional function with a global minimum of 0 at (1,10,1,5)

    Parameters:
        x: (x1, x2, x3, x4) this is a 5D problem
    version : str
        The version to use for the function's computation.
        Options are 'numpy' and 'pytorch'.

    Returns:
    result : np.ndarray or torch.Tensor
        The computed Damavandi function values
        corresponding to the inputs (x1, x2, x3, x4).

    Raises:
    ValueError
        If the version is not 'numpy' or 'pytorch'.
    """
    x1, x2, x3, x4 = x.flatten()
    if version == 'numpy':
        t = np.arange(1, 11) * 0.1
        y = np.exp(-t) - 5 * np.exp(10 * t)
        result = np.sum(np.square(x3 * np.exp(-t * x1) - x4 * np.exp(-t * x2) - y))
    elif version == 'pytorch':
        t = torch.arange(1, 11) * 0.1
        y = torch.exp(-t) - 5 * torch.exp(10 * t)
        result = torch.sum(
            torch.square(x3 * torch.exp(-t * x1) - x4 * torch.exp(-t * x2) - y)
        )
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
                x3.detach().cpu().numpy(),
                x4.detach().cpu().numpy(),
                result.detach().cpu().numpy(),
            )
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, x3, x4, result))

    return result


# Biggs EXP5
def biggs_exp5(x, results, trial, version='numpy'):
    """
    Implementation of the Biggs EXP5 function.
    This is a 6-dimensional function with a global minimum of 0 at (1,10,1,5,4)

    Parameters:
        x: (x1, x2, x3, x4, x5) this is a 6D problem
    version : str
        The version to use for the function's computation.
        Options are 'numpy' and 'pytorch'.

    Returns:
    result : np.ndarray or torch.Tensor
        The computed Damavandi function values
        corresponding to the inputs (x1, x2, x3, x4, x5).

    Raises:
    ValueError
        If the version is not 'numpy' or 'pytorch'.
    """
    x1, x2, x3, x4, x5 = x.flatten()
    if version == 'numpy':
        t = np.arange(1, 11) * 0.1
        y = np.exp(-t) - 5 * np.exp(10 * t) + 3 * np.exp(-4 * t)
        result = np.sum(
            np.square(
                x3 * np.exp(-t * x1) - x4 * np.exp(-t * x2) + 3 * np.exp(-t * x5) - y
            )
        )
    elif version == 'pytorch':
        t = torch.arange(1, 11) * 0.1
        y = torch.exp(-t) - 5 * torch.exp(10 * t) + 3 * torch.exp(-4 * t)
        result = torch.sum(
            torch.square(
                x3 * torch.exp(-t * x1)
                - x4 * torch.exp(-t * x2)
                + 3 * torch.exp(-t * x5)
                - y
            )
        )
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
                x3.detach().cpu().numpy(),
                x4.detach().cpu().numpy(),
                x5.detach().cpu().numpy(),
                result.detach().cpu().numpy(),
            )
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, x3, x4, x5, result))

    return result


# Biggs EXP6
def biggs_exp6(x, results, trial, version='numpy'):
    """
    Implementation of the Biggs EXP5 function.
    This is a 7-dimensional function with a global minimum of 0 at (1,10,1,5,4,3)

    Parameters:
        x: (x1, x2, x3, x4, x5, x6) this is a 7D problem
    version : str
        The version to use for the function's computation.
        Options are 'numpy' and 'pytorch'.

    Returns:
    result : np.ndarray or torch.Tensor
        The computed Damavandi function values
        corresponding to the inputs (x1, x2, x3, x4, x5, x6).

    Raises:
    ValueError
        If the version is not 'numpy' or 'pytorch'.
    """
    x1, x2, x3, x4, x5, x6 = x.flatten()
    if version == 'numpy':
        t = np.arange(1, 11) * 0.1
        y = np.exp(-t) - 5 * np.exp(10 * t) + 3 * np.exp(-4 * t)
        result = np.sum(
            np.square(
                x3 * np.exp(-t * x1) - x4 * np.exp(-t * x2) + x6 * np.exp(-t * x5) - y
            )
        )
    elif version == 'pytorch':
        t = torch.arange(1, 11) * 0.1
        y = torch.exp(-t) - 5 * torch.exp(10 * t) + 3 * torch.exp(-4 * t)
        result = torch.sum(
            torch.square(
                x3 * torch.exp(-t * x1)
                - x4 * torch.exp(-t * x2)
                + x6 * torch.exp(-t * x5)
                - y
            )
        )
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
                x3.detach().cpu().numpy(),
                x4.detach().cpu().numpy(),
                x5.detach().cpu().numpy(),
                x6.detach().cpu().numpy(),
                result.detach().cpu().numpy(),
            )
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, x3, x4, x5, x6, result))

    return result


# Bird in 2d
def bird(x, results, trial, version='numpy'):
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = (
            np.sin(x1) * np.exp(np.square(1 - np.cos(x2)))
            + np.cos(x2) * np.exp(np.square(1 - np.sin(x1)))
            + np.square(x1 - x2)
        )
    elif version == 'pytorch':
        result = (
            torch.sin(x1) * torch.exp(torch.square(1 - torch.cos(x2)))
            + torch.cos(x2) * torch.exp(torch.square(1 - torch.sin(x1)))
            + torch.square(x1 - x2)
        )
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


# Bohachevsky1 in 2d
def bohachevsky1(x, results, trial, version='numpy'):
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = (
            np.square(x1)
            + 2 * np.square(x2)
            - 0.3 * np.cos(3 * np.pi * x1)
            - 0.4 * np.cos(4 * np.pi * x2)
            + 0.7
        )
    elif version == 'pytorch':
        result = (
            torch.square(x1)
            + 2 * torch.square(x2)
            - 0.3 * torch.cos(3 * np.pi * x1)
            - 0.4 * torch.cos(4 * np.pi * x2)
            + 0.7
        )
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


# Bohachevsky2 in 2d
def bohachevsky2(x, results, trial, version='numpy'):
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = (
            np.square(x1)
            + 2 * np.square(x2)
            - 0.3 * np.cos(3 * np.pi * x1) * 0.4 * np.cos(4 * np.pi * x2)
            + 0.3
        )
    elif version == 'pytorch':
        result = (
            torch.square(x1)
            + 2 * torch.square(x2)
            - 0.3 * torch.cos(3 * np.pi * x1) * 0.4 * torch.cos(4 * np.pi * x2)
            + 0.3
        )
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


# Bohachevsky3 in 2d
def bohachevsky3(x, results, trial, version='numpy'):
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = (
            np.square(x1)
            + 2 * np.square(x2)
            - 0.3 * np.cos(3 * np.pi * x1 + 4 * np.pi * x2)
            + 0.3
        )
    elif version == 'pytorch':
        result = (
            torch.square(x1)
            + 2 * torch.square(x2)
            - 0.3 * torch.cos(3 * np.pi * x1 + 4 * np.pi * x2)
            + 0.3
        )
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


# Booth in 2d
def booth(x, results, trial, version='numpy'):
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = np.square(x1 + 2 * x2 - 7) + np.square(2 * x1 + x2 - 5)
    elif version == 'pytorch':
        result = torch.square(x1 + 2 * x2 - 7) + torch.square(2 * x1 + x2 - 5)
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


# Branin RCOS in 2d
def branin_rcos(x, results, trial, version='numpy'):
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = (
            np.square(x2 - ((5.1 * np.square(x1)) / (4 * np.square(np.pi))) - 6)
            + 10
            * (1 - (1 / (8 * np.pi)))
            * np.cos(x1)
            * np.cos(x2)
            * np.log(np.square(x1) + np.square(x2) + 1)
            + 10
        )
    elif version == 'pytorch':
        result = (
            torch.square(x2 - ((5.1 * torch.square(x1)) / (4 * torch.pi**2)) - 6)
            + 10
            * (1 - (1 / (8 * np.pi)))
            * torch.cos(x1)
            * torch.cos(x2)
            * torch.log(torch.square(x1) + torch.square(x2) + 1)
            + 10
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


# Brent function in 2d
def brent(x, results, trial, version='numpy'):
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = (
            np.square(x1 + 10)
            + np.square(x2 + 10)
            + np.exp(-np.square(x1) - np.square(x2))
        )
    elif version == 'pytorch':
        result = (
            torch.square(x1 + 10)
            + torch.square(x2 + 10)
            + torch.exp(-torch.square(x1) - torch.square(x2))
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


# nd Brown function
def brown(x, results, trial, version='numpy'):
    x = x.flatten()
    shifted_x = x[1:]
    x = x[:-1]
    if version == 'numpy':
        result = np.sum(
            np.power(np.square(x), np.square(shifted_x) + 1)
            + np.power(np.square(shifted_x), np.square(shifted_x) + 1)
        )
    elif version == 'pytorch':
        result = torch.sum(
            torch.pow(torch.square(x), torch.square(shifted_x) + 1)
            + torch.pow(torch.square(shifted_x), torch.square(shifted_x) + 1)
        )
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

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


# Bukin n2 in 2d
def bukin_n2(x, results, trial, version='numpy'):
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = 100 * (x2 - 0.01 * np.square(x1) + 1) + 0.01 * np.square(x1 + 10)
    elif version == 'pytorch':
        result = 100 * (x2 - 0.01 * torch.square(x1) + 1) + 0.01 * torch.square(x1 + 10)
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


# Bukin n4 in 2d
def bukin_n4(x, results, trial, version):
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = 100 * np.square(x2) + 0.01 * np.abs(x1 + 10)
    elif version == 'pytorch':
        result = 100 * torch.square(x2) + 0.01 * torch.abs(x1 + 10)
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


# Camel 3 hump in 2d
def camel_3hump(x, results, trial, version='numpy'):
    x1, x2 = x.flatten()
    if version == 'numpy' or version == 'pytorch':
        result = 2 * x1**2 - 1.05 * x1**4 + (1 / 6) * x1**6 + x1 * x2 + x2**2
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


# Camel 6 hump in 2d
def camel_6hump(x, results, trial, version='numpy'):
    x1, x2 = x.flatten()
    if version == 'numpy' or version == 'pytorch':
        result = (
            (4 - 2.1 * x1**2 + (1 / 3) * x1**4) * x1**2
            + x1 * x2
            + (4 * x2**2 - 4) * x2**2
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


# Chen Bird in 2d
def chen_bird(x, results, trial, version='numpy'):
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = -(0.001 / np.floor(0.001**2 + (x1 - 0.4 * x2 - 0.1) ** 2)) - (
            0.001 / np.floor(0.001**2 + (2 * x1 + x2 - 1.5) ** 2)
        )
    elif version == 'pytorch':
        result = -(0.001 / torch.floor(0.001**2 + (x1 - 0.4 * x2 - 0.1) ** 2)) - (
            0.001 / torch.floor(0.001**2 + (2 * x1 + x2 - 1.5) ** 2)
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


# Chen V in 2d
def chen_v(x, results, trial, version='numpy'):
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = (
            -(0.001 / np.floor(0.001**2 + (x1**2 + x2**2 - 1) ** 2))
            - (0.001 / np.floor(0.001**2 + (x1**2 + x2**2 - 0.5) ** 2))
            - (0.001 / np.floor(0.001**2 + (x1**2 - x2**2) ** 2))
        )
    elif version == 'pytorch':
        result = (
            -(0.001 / torch.floor(0.001**2 + (x1**2 + x2**2 - 1) ** 2))
            - (0.001 / torch.floor(0.001**2 + (x1**2 + x2**2 - 0.5) ** 2))
            - (0.001 / torch.floor(0.001**2 + (x1**2 - x2**2) ** 2))
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


# Chichinadze in 2d
def chichinadze(x, results, trial, version='numpy'):
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = (
            x1**2
            - 12 * x1
            + 11
            + 10 * np.cos((np.pi * x1) / 2)
            + 8 * np.sin((5 * np.pi * x1) / 2)
            - (1 / 5) ** 0.5 * np.exp(-0.5 * (x2 - 0.5) ** 2)
        )
    elif version == 'pytorch':
        result = (
            x1**2
            - 12 * x1
            + 11
            + 10 * torch.cos((torch.pi * x1) / 2)
            + 8 * torch.sin((5 * torch.pi * x1) / 2)
            - (1 / 5) ** 0.5 * torch.exp(-0.5 * (x2 - 0.5) ** 2)
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


# nd Chung Reynolds
def chung_reynolds(x, results, trial, version='numpy'):
    x = x.flatten()
    if version == 'numpy':
        result = np.square(np.sum(np.square(x)))
    elif version == 'pytorch':
        result = torch.square(torch.sum(torch.square(x)))
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

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


# Colville in 4d
def colville(x, results, trial, version='numpy'):
    x1, x2, x3, x4 = x.flatten()
    if version == 'numpy' or version == 'pytorch':
        result = (
            100 * (x1 - x2**2) ** 2
            + (1 - x1) ** 2
            + 90 * (x4 - x3**2) ** 2
            + (1 - x3) ** 2
            + 10.1 * ((x2 - 1) ** 2 + (x4 - 1) ** 2)
            + 19.8 * (x2 - 1) * (x4 - 1)
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
                result.detach().cpu().numpy(),
            )
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, x3, x4, result))

    return result


# Cosine Mixture in 4d
def cosine_mixture(x, results, trial, version='numpy'):
    x = x.flatten()
    if version == 'numpy':
        result = -0.1 * np.sum(np.cos(5 * np.pi * x)) - np.sum(np.square(x))
    elif version == 'pytorch':
        result = -0.1 * torch.sum(torch.cos(5 * np.pi * x)) - torch.sum(torch.square(x))
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

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


# nd Csendes fn
def csendes(x, results, trial, version='numpy'):
    x = x.flatten()
    if version == 'numpy':
        result = np.sum(np.power(x, 6) * (2 + np.sin(1 / x)))
    elif version == 'pytorch':
        result = torch.sum(torch.pow(x, 6) * (2 + torch.sin(1 / x)))
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

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


# Cube in 2d
def cube(x, results, trial, version='numpy'):
    x1, x2 = x.flatten()
    if version == 'numpy' or version == 'pytorch':
        result = 100 * (x2 - x1**3) ** 2 + (1 - x1) ** 2
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


# nd Deb 1 fn
def deb1(x, results, trial, version='numpy'):
    x = x.flatten()
    d = len(x)
    if version == 'numpy':
        result = -(1 / d) * np.sum(np.power(np.sin(5 * np.pi * x), 6))
    elif version == 'pytorch':
        result = -(1 / d) * torch.sum(torch.pow(torch.sin(5 * torch.pi * x), 6))
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

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


# nd Deb 3 fn
def deb3(x, results, trial, version='numpy'):
    x = x.flatten()
    d = len(x)
    if version == 'numpy':
        result = -(1 / d) * np.sum(
            np.power(np.sin(5 * np.pi * (np.power(x, 0.75) - 0.05)), 6)
        )
    elif version == 'pytorch':
        result = -(1 / d) * torch.sum(
            torch.pow(torch.sin(5 * torch.pi * (torch.pow(x, 0.75) - 0.05)), 6)
        )
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

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


# Deckkers-Aarts in 2d
def deckkers_aarts(x, results, trial, version='numpy'):
    x1, x2 = x.flatten()
    if version == 'numpy' or version == 'pytorch':
        result = (
            (10**5) * (x1**2)
            + x2**2
            - (x1**2 + x2**2) ** 2
            + (10 ** (-5)) * (x1**2 + x2**2) ** 4
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


# deVilliers Glasser 1 in 4d
def devilliers_glasser1(x, results, trial, version='numpy'):
    x1, x2, x3, x4 = x.flatten()
    if version == 'numpy':
        t = 0.1 * np.arange(0, 24)
        y = 60.137 * np.power(1.371, t) * np.sin(3.112 * t + 1.761)
        result = np.sum(np.square(x1 * np.power(x2, t) * np.sin(x3 * t + x4) - y))
    elif version == 'pytorch':
        t = 0.1 * torch.arange(0, 24)
        y = 60.137 * torch.pow(1.371, t) * torch.sin(3.112 * t + 1.761)
        result = torch.sum(
            torch.square(x1 * torch.pow(x2, t) * torch.sin(x3 * t + x4) - y)
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
                result.detach().cpu().numpy(),
            )
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, x3, x4, result))

    return result


# deVilliers Glasser 2 in 5d
def devilliers_glasser2(x, results, trial, version='numpy'):
    x1, x2, x3, x4, x5 = x.flatten()
    if version == 'numpy':
        t = 0.1 * np.arange(0, 16)
        y = (
            53.81
            * np.power(1.27, t)
            * np.tanh(3.012 * t + np.sin(2.13 * t))
            * np.cos(np.exp(0.507) * t)
        )
        result = np.sum(
            np.square(
                x1
                * np.power(x2, t)
                * np.tanh(x3 * t + np.sin(x4 * t))
                * np.cos(t * np.exp(x5))
                - y
            )
        )
    elif version == 'pytorch':
        t = 0.1 * torch.arange(0, 16)
        y = (
            53.81
            * torch.pow(1.27, t)
            * torch.tanh(3.012 * t + torch.sin(2.13 * t))
            * torch.cos(torch.exp(0.507) * t)
        )
        result = torch.sum(
            torch.square(
                x1
                * torch.pow(x2, t)
                * torch.tanh(x3 * t + torch.sin(x4 * t))
                * torch.cos(t * torch.exp(x5))
                - y
            )
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
                result.detach().cpu().numpy(),
            )
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, x3, x4, x5, result))

    return result


# nd Dixon and Price
def dixon_price(x, results, trial, version='numpy'):
    x = x.flatten()
    d = len(x)
    x1 = x[0]
    shifted_x = x[:-1]
    x = x[1:]
    if version == 'numpy':
        i = np.arange(2, d)
        result = np.square(x1 - 1) + np.sum(i * np.square(2 * np.square(x) - shifted_x))
    elif version == 'pytorch':
        i = torch.arange(2, d)
        result = torch.square(x1 - 1) + torch.sum(
            i * torch.square(2 * torch.square(x) - shifted_x)
        )
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

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


# Dolan fn in 5d
def dolan(x, results, trial, version='numpy'):
    x1, x2, x3, x4, x5 = x.flatten()
    if version == 'numpy':
        result = (
            (x1 + 1.7 * x2) * np.sin(x1)
            - 1.5 * x3
            - 0.1 * x4 * np.cos(x4 + x5 - x1)
            + 0.2 * x5**2
            - x2
            - 1
        )
    elif version == 'pytorch':
        result = (
            (x1 + 1.7 * x2) * torch.sin(x1)
            - 1.5 * x3
            - 0.1 * x4 * torch.cos(x4 + x5 - x1)
            + 0.2 * x5**2
            - x2
            - 1
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
                result.detach().cpu().numpy(),
            )
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, x3, x4, x5, result))

    return result


# Easom in 2d
def easom(x, results, trial, version='numpy'):
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = (
            -np.cos(x1)
            * np.cos(x2)
            * np.exp(-np.square(x1 - np.pi) - np.square(x2 - np.pi))
        )
    elif version == 'pytorch':
        result = (
            -torch.cos(x1)
            * torch.cos(x2)
            * torch.exp(-torch.square(x1 - torch.pi) - torch.square(x2 - torch.pi))
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


# El-Attar-Vidysagar-Dutta fn in 2d
def el_attar(x, results, trial, version='numpy'):
    x1, x2 = x.flatten()
    if version == 'numpy' or version == 'pytorch':
        result = (
            (x1**2 + x2 - 10) ** 2
            + (x1 + x2**2 - 7) ** 2
            + (x1**2 + x2**3 - 1) ** 2
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


# Egg Crate fn in 2d
def egg_crate(x, results, trial, version='numpy'):
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = (
            x1**2 + x2**2 + 25 * (np.square(np.sin(x1)) + np.square(np.sin(x2)))
        )
    elif version == 'pytorch':
        result = (
            x1**2
            + x2**2
            + 25 * (torch.square(torch.sin(x1)) + torch.square(torch.sin(x2)))
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


# nd Exponenetial fn
def exp1(x, results, trial, version='numpy'):
    x = x.flatten()
    if version == 'numpy':
        result = -np.exp(-0.5 * np.sum(np.square(x)))
    elif version == 'pytorch':
        result = -torch.exp(-0.5 * torch.sum(torch.square(x)))
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

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


# Exp2 fn in 2d
def exp2(x, results, trial, version='numpy'):
    x1, x2 = x.flatten()
    if version == 'numpy':
        i = np.arange(0, 10)
        result = np.sum(
            np.square(
                np.exp((-i * x1) / 10)
                - 5 * np.exp((-i * x2) / 10)
                - np.exp(-i / 10)
                + 5 * np.exp(-i)
            )
        )
    elif version == 'pytorch':
        i = torch.arange(0, 10)
        result = torch.sum(
            torch.square(
                torch.exp((-i * x1) / 10)
                - 5 * torch.exp((-i * x2) / 10)
                - torch.exp(-i / 10)
                + 5 * torch.exp(-i)
            )
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


# Freudenstein Roth fn in 2d
def freudenstein_roth(x, results, trial, version='numpy'):
    x1, x2 = x.flatten()
    if version == 'numpy' or version == 'pytorch':
        result = (x1 - 13 + ((5 - x2) * x2 - 2) * x2) ** 2 + (
            x1 - 29 + ((x2 + 1) * x2 - 14) * x2
        ) ** 2
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


# Giunta in 2d
def giunta(x, results, trial, version='numpy'):
    x1, x2 = x.flatten()
    if version == 'numpy':
        term1 = (
            np.sin((16 - 15) * x1 - 1)
            + np.square(np.sin((16 / 15) * x1 - 1))
            + (1 / 50) * np.sin(4 * ((16 / 15) * x1 - 1))
        )
        term2 = (
            np.sin((16 - 15) * x2 - 1)
            + np.square(np.sin((16 / 15) * x2 - 1))
            + (1 / 50) * np.sin(4 * ((16 / 15) * x2 - 1))
        )
        result = 0.6 + term1 + term2
    elif version == 'pytorch':
        term1 = (
            torch.sin((16 - 15) * x1 - 1)
            + torch.square(torch.sin((16 / 15) * x1 - 1))
            + (1 / 50) * torch.sin(4 * ((16 / 15) * x1 - 1))
        )
        term2 = (
            torch.sin((16 - 15) * x2 - 1)
            + torch.square(torch.sin((16 / 15) * x2 - 1))
            + (1 / 50) * torch.sin(4 * ((16 / 15) * x2 - 1))
        )
        result = 0.6 + term1 + term2
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


# Goldstein Price fn in 2d
def goldstein_price(x, results, trial, version='numpy'):
    x1, x2 = x.flatten()
    if version == 'numpy' or version == 'pytorch':
        term1 = 1 + (x1 + x2 + 1) ** 2 * (
            19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2
        )
        term2 = 30 + (2 * x1 - 3 * x2) ** 2 * (
            18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2
        )
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


# # Documentation for this problem is weird
# # Hansen in 2d
# def hansen(x, results, trial, version='numpy'):
#     x1, x2 = x.flatten()
#     if version == 'numpy':
#         i = np.arange(0, 4)
#         j = np.arange(0, 4)
#         result = np.sum((i+1)*np.cos(i*x1+i+1)) * np.sum

# Hartman3 in 3d

# Hartman6 in 6d


# Helical Valley in 3d
def helical_valley(x, results, trial, version='numpy'):
    x1, x2, x3 = x.flatten()
    if version == 'numpy':
        if x1 >= 0:
            theta = (1 / (2 * np.pi)) * np.arctan(x1 / x2)
        else:
            theta = (1 / (2 * np.pi)) * np.arctan(x1 / x2 + 0.5)
        result = 100 * (
            np.square(x2 - 10 * theta) + np.square(np.sqrt(x1**2 + x2**2) - 1)
        ) + np.square(x3)
    elif version == 'pytorch':
        if x1 >= 0:
            theta = (1 / (2 * torch.pi)) * torch.atan(x1 / x2)
        else:
            theta = (1 / (2 * torch.pi)) * torch.atan(x1 / x2 + 0.5)
        result = 100 * (
            torch.square(x2 - 10 * theta)
            + torch.square(torch.sqrt(x1**2 + x2**2) - 1)
        ) + torch.square(x3)
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


# Himmelblau fn in 2d
def himmelblau(x, results, trial, version='numpy'):
    x1, x2 = x.flatten()
    if version == 'numpy' or version == 'pytorch':
        result = (x1**2 + x2 - 11) ** 2 + (x1 + x2**2 - 7) ** 2
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


# Hosaki fn in 2d
def hosaki(x, results, trial, version='numpy'):
    x1, x2 = x.flatten()
    if version == 'numpy':
        result = (
            (1 - 8 * x1 + 7 * x1**2 - (7 / 3) * x1**3 + (1 / 4) * x1**4)
            * x2**2
            * np.exp(-x2)
        )
    elif version == 'pytorch':
        result = (
            (1 - 8 * x1 + 7 * x1**2 - (7 / 3) * x1**3 + (1 / 4) * x1**4)
            * x2**2
            * torch.exp(-x2)
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


# Jennrich-Sampson fn in 2d
def jennrich_sampson(x, results, trial, version='numpy'):
    x1, x2 = x.flatten()
    if version == 'numpy':
        i = np.arange(1, 11)
        result = np.sum(np.square(2 + 2 * i - (np.exp(i * x1) + np.exp(i * x2))))
    elif version == 'pytorch':
        i = np.arange(1, 11)
        result = torch.sum(
            torch.square(2 + 2 * i - (torch.exp(i * x1) + torch.exp(i * x2)))
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

# Langermann
langermann_config = {
    'objective': langermann,
    'bounds': [(0, 10), (0, 10)],
    'max_iterations': 1000,
    'global_minimum': None,
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
    'max_iterations': 100,
    'global_minimum': 0.0,
    'dimensions': 2,
}

# Schaffer N4
schaffer_n4_config = {
    'objective': schaffer_n4,
    'bounds': [(-100.0, 100.0), (-100.0, 100.0)],
    'max_iterations': 1000,
    'global_minimum': 0.292579,
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
    'bounds': [(-10, 10), (-10, 10)],
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

ackley_100d_config = {
    'objective': ndackley,
    'bounds': [(-32.768, 32.768)],  # Will use a single level bound and then expand
    'max_iterations': 1000,
    'global_minimum': 0.0,
    'dimensions': 100,
}

ackley_1000d_config = {
    'objective': ndackley,
    'bounds': [(-32.768, 32.768)],  # Will use a single level bound and then expand
    'max_iterations': 1000,
    'global_minimum': 0.0,
    'dimensions': 1000,
}

# Multi-Dimensional Problems #
griewank_3d_config = {
    'objective': ndgriewank,
    'bounds': [(-600, 600)],  # Will use a single level bound and then expand
    'max_iterations': 1000,
    'global_minimum': 0.0,
    'dimensions': 3,
}

# Multi-Dimensional Problems #
griewank_5d_config = {
    'objective': ndgriewank,
    'bounds': [(-600, 600)],  # Will use a single level bound and then expand
    'max_iterations': 1000,
    'global_minimum': 0.0,
    'dimensions': 5,
}

# Multi-Dimensional Problems #
griewank_30d_config = {
    'objective': ndgriewank,
    'bounds': [(-600, 600)],  # Will use a single level bound and then expand
    'max_iterations': 1000,
    'global_minimum': 0.0,
    'dimensions': 30,
}

griewank_100d_config = {
    'objective': ndgriewank,
    'bounds': [(-600, 600)],  # Will use a single level bound and then expand
    'max_iterations': 1000,
    'global_minimum': 0.0,
    'dimensions': 100,
}

griewank_1000d_config = {
    'objective': ndgriewank,
    'bounds': [(-600, 600)],  # Will use a single level bound and then expand
    'max_iterations': 1000,
    'global_minimum': 0.0,
    'dimensions': 1000,
}

# Multi-Dimensional Problems #
levy_3d_config = {
    'objective': ndlevy,
    'bounds': [(-10, 10)],  # Will use a single level bound and then expand
    'max_iterations': 1000,
    'global_minimum': 0.0,
    'dimensions': 3,
}

# Multi-Dimensional Problems #
levy_5d_config = {
    'objective': ndlevy,
    'bounds': [(-10, 10)],  # Will use a single level bound and then expand
    'max_iterations': 1000,
    'global_minimum': 0.0,
    'dimensions': 5,
}

# Multi-Dimensional Problems #
levy_30d_config = {
    'objective': ndlevy,
    'bounds': [(-10, 10)],  # Will use a single level bound and then expand
    'max_iterations': 1000,
    'global_minimum': 0.0,
    'dimensions': 30,
}

# Multi-Dimensional Problems #
rastrigin_3d_config = {
    'objective': ndrastrigin,
    'bounds': [(-5.12, 5.12)],  # Will use a single level bound and then expand
    'max_iterations': 1000,
    'global_minimum': 0.0,
    'dimensions': 3,
}

rastrigin_5d_config = {
    'objective': ndrastrigin,
    'bounds': [(-5.12, 5.12)],  # Will use a single level bound and then expand
    'max_iterations': 1000,
    'global_minimum': 0.0,
    'dimensions': 5,
}

rastrigin_30d_config = {
    'objective': ndrastrigin,
    'bounds': [(-5.12, 5.12)],  # Will use a single level bound and then expand
    'max_iterations': 1000,
    'global_minimum': 0.0,
    'dimensions': 30,
}


schwefel_3d_config = {
    'objective': ndschwefel,
    'bounds': [(-500, 500)],  # Will use a single level bound and then expand
    'max_iterations': 1000,
    'global_minimum': 0.0,
    'dimensions': 3,
}

schwefel_5d_config = {
    'objective': ndschwefel,
    'bounds': [(-500, 500)],  # Will use a single level bound and then expand
    'max_iterations': 1000,
    'global_minimum': 0.0,
    'dimensions': 5,
}

schwefel_30d_config = {
    'objective': ndschwefel,
    'bounds': [(-500, 500)],  # Will use a single level bound and then expand
    'max_iterations': 1000,
    'global_minimum': 0.0,
    'dimensions': 30,
}

# ex8_6_2 from MINLP
ex8_6_2_config = {
    'objective': ex8_6_2,
    'bounds': [
        (-1e-8, 1e-8),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-1e-8, 1e-8),
        (-1e-8, 1e-8),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-1e-8, 1e-8),
        (-1e-8, 1e-8),
        (-1e-8, 1e-8),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
        (-5, 5),
    ],
    'max_iterations': 1000,
    'global_minimum': -31.888630,
    'dimensions': 30,
}

# least from MINLP
least_config = {
    'objective': least,
    'bounds': [
        (-1e20, 1e20),
        (-1e20, 1e20),
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
    'global_minimum': 0,
    'dimensions': 2,
}

damavandi_config = {
    'objective': damavandi,
    'bounds': [
        (0, 14),
        (0, 14),
    ],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 2,
}

cross_leg_table_config = {
    'objective': cross_leg_table,
    'bounds': [
        (-10, 10),
        (-10, 10),
    ],
    'max_iterations': 1000,
    'global_minimum': -1,
    'dimensions': 2,
}

sine_envelope_config = {
    'objective': sine_envelope,
    'bounds': [
        (-100, 100),
        (-100, 100),
    ],
    'max_iterations': 1000,
    'global_minimum': -0.72984,
    'dimensions': 2,
}

ackley2_config = {
    'objective': ackley2,
    'bounds': [
        (-32, 32),
        (-32, 32),
    ],
    'max_iterations': 1000,
    'global_minimum': -200,
    'dimensions': 2,
}

ackley3_config = {
    'objective': ackley3,
    'bounds': [
        (-32, 32),
        (-32, 32),
    ],
    'max_iterations': 1000,
    'global_minimum': -219.1418,
    'dimensions': 2,
}

ackley4_config = {
    'objective': ndackley4,
    'bounds': [(-35, 35), (-35, 35)],
    'max_iterations': 1000,
    'global_minimum': -3.917275,
    'dimensions': 2,
}

ackley4_10d_config = {
    'objective': ndackley4,
    'bounds': [(-35, 35)],
    'max_iterations': 1000,
    'global_minimum': -3.917275,
    'dimensions': 10,
}

ackley4_50d_config = {
    'objective': ndackley4,
    'bounds': [(-35, 35)],
    'max_iterations': 1000,
    'global_minimum': -3.917275,
    'dimensions': 50,
}

ackley4_100d_config = {
    'objective': ndackley4,
    'bounds': [(-35, 35)],
    'max_iterations': 1000,
    'global_minimum': -3.917275,
    'dimensions': 100,
}

ackley4_500d_config = {
    'objective': ndackley4,
    'bounds': [(-35, 35)],
    'max_iterations': 1000,
    'global_minimum': -3.917275,
    'dimensions': 500,
}

ackley4_1000d_config = {
    'objective': ndackley4,
    'bounds': [(-35, 35)],
    'max_iterations': 1000,
    'global_minimum': -3.917275,
    'dimensions': 1000,
}

ackley4_5000d_config = {
    'objective': ndackley4,
    'bounds': [(-35, 35)],
    'max_iterations': 1000,
    'global_minimum': -3.917275,
    'dimensions': 5000,
}

adjiman_config = {
    'objective': adjiman,
    'bounds': [
        (-1, 2),
        (-1, 1),
    ],
    'max_iterations': 1000,
    'global_minimum': -2.02181,
    'dimensions': 2,
}

alpine1_config = {
    'objective': alpine1,
    'bounds': [(-10, 10), (-10, 10)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 2,
}

alpine1_10d_config = {
    'objective': alpine1,
    'bounds': [(-10, 10)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 10,
}

alpine1_50d_config = {
    'objective': alpine1,
    'bounds': [(-10, 10)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 50,
}

alpine1_100d_config = {
    'objective': alpine1,
    'bounds': [(-10, 10)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 100,
}

alpine1_500d_config = {
    'objective': alpine1,
    'bounds': [(-10, 10)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 500,
}

alpine1_1000d_config = {
    'objective': alpine1,
    'bounds': [(-10, 10)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 1000,
}

alpine1_5000d_config = {
    'objective': alpine1,
    'bounds': [(-10, 10)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 5000,
}

alpine2_config = {
    'objective': alpine2,
    'bounds': [(0, 10), (0, 10)],
    'max_iterations': 1000,
    'global_minimum': 2.808**2,
    'dimensions': 2,
}

alpine2_10d_config = {
    'objective': alpine2,
    'bounds': [(0, 10)],
    'max_iterations': 1000,
    'global_minimum': 2.808**10,
    'dimensions': 10,
}

brad_config = {
    'objective': brad,
    'bounds': [
        (-0.25, 0.25),
        (0.01, 1e10),
        (-1e10, 2.5),
    ],
    'max_iterations': 1000,
    'global_minimum': 0.00821487,
    'dimensions': 3,
}

bartels_conn_config = {
    'objective': bartels_conn,
    'bounds': [
        (-500, 500),
        (-500, 500),
    ],
    'max_iterations': 1000,
    'global_minimum': 1,
    'dimensions': 2,
}

beale_config = {
    'objective': beale,
    'bounds': [
        (-4.5, 4.5),
        (-4.5, 4.5),
    ],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 2,
}

biggs_exp2_config = {
    'objective': biggs_exp2,
    'bounds': [
        (0, 20),
        (0, 20),
    ],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 2,
}

biggs_exp3_config = {
    'objective': biggs_exp3,
    'bounds': [
        (0, 20),
        (0, 20),
        (0, 20),
    ],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 3,
}

biggs_exp4_config = {
    'objective': biggs_exp4,
    'bounds': [
        (0, 20),
        (0, 20),
        (0, 20),
        (0, 20),
    ],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 4,
}

biggs_exp5_config = {
    'objective': biggs_exp5,
    'bounds': [
        (0, 20),
        (0, 20),
        (0, 20),
        (0, 20),
        (0, 20),
    ],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 5,
}

biggs_exp6_config = {
    'objective': biggs_exp6,
    'bounds': [
        (0, 20),
        (0, 20),
        (0, 20),
        (0, 20),
        (0, 20),
    ],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 6,
}

bird_config = {
    'objective': bird,
    'bounds': [
        (-2 * np.pi, 2 * np.pi),
        (-2 * np.pi, 2 * np.pi),
    ],
    'max_iterations': 1000,
    'global_minimum': -106.764537,
    'dimensions': 2,
}

bohachevsky1_config = {
    'objective': bohachevsky1,
    'bounds': [
        (-100, 100),
        (-100, 100),
    ],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 2,
}

bohachevsky2_config = {
    'objective': bohachevsky2,
    'bounds': [
        (-100, 100),
        (-100, 100),
    ],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 2,
}

bohachevsky3_config = {
    'objective': bohachevsky3,
    'bounds': [
        (-100, 100),
        (-100, 100),
    ],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 2,
}

booth_config = {
    'objective': booth,
    'bounds': [
        (-10, 10),
        (-10, 10),
    ],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 2,
}

branin_rcos_config = {
    'objective': branin_rcos,
    'bounds': [
        (-5, 15),
        (-5, 15),
    ],
    'max_iterations': 1000,
    'global_minimum': 5.559037,
    'dimensions': 2,
}

brent_config = {
    'objective': brent,
    'bounds': [
        (-10, 10),
        (-10, 10),
    ],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 2,
}

brown_config = {
    'objective': brown,
    'bounds': [(-1, 4), (-1, 4)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 2,
}

brown_10d_config = {
    'objective': brown,
    'bounds': [(-1, 4)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 10,
}

brown_50d_config = {
    'objective': brown,
    'bounds': [(-1, 4)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 50,
}

brown_100d_config = {
    'objective': brown,
    'bounds': [(-1, 4)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 100,
}

brown_500d_config = {
    'objective': brown,
    'bounds': [(-1, 4)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 500,
}

brown_1000d_config = {
    'objective': brown,
    'bounds': [(-1, 4)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 1000,
}

brown_5000d_config = {
    'objective': brown,
    'bounds': [(-1, 4)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 5000,
}

bukin_n2_config = {
    'objective': bukin_n2,
    'bounds': [
        (-15, -5),
        (-3, 3),
    ],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 2,
}

bukin_n4_config = {
    'objective': bukin_n4,
    'bounds': [
        (-15, -5),
        (-3, 3),
    ],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 2,
}

camel_3hump_config = {
    'objective': camel_3hump,
    'bounds': [
        (-5, 5),
        (-5, 5),
    ],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 2,
}

camel_6hump_config = {
    'objective': camel_6hump,
    'bounds': [
        (-5, 5),
        (-5, 5),
    ],
    'max_iterations': 1000,
    'global_minimum': -1.0316,
    'dimensions': 2,
}

chen_bird_config = {
    'objective': chen_bird,
    'bounds': [
        (-500, 500),
        (-500, 500),
    ],
    'max_iterations': 1000,
    'global_minimum': -2000,
    'dimensions': 2,
}

chen_v_config = {
    'objective': chen_v,
    'bounds': [
        (-500, 500),
        (-500, 500),
    ],
    'max_iterations': 1000,
    'global_minimum': -2000,
    'dimensions': 2,
}

chichinadze_config = {
    'objective': chichinadze,
    'bounds': [
        (-30, 30),
        (-30, 30),
    ],
    'max_iterations': 1000,
    'global_minimum': -43.3159,
    'dimensions': 2,
}

chung_reynolds_config = {
    'objective': chung_reynolds,
    'bounds': [(-100, 100), (-100, 100)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 2,
}

chung_reynolds_10d_config = {
    'objective': chung_reynolds,
    'bounds': [(-100, 100)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 10,
}

chung_reynolds_50d_config = {
    'objective': chung_reynolds,
    'bounds': [(-100, 100)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 50,
}

chung_reynolds_100d_config = {
    'objective': chung_reynolds,
    'bounds': [(-100, 100)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 100,
}

chung_reynolds_500d_config = {
    'objective': chung_reynolds,
    'bounds': [(-100, 100)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 500,
}

chung_reynolds_1000d_config = {
    'objective': chung_reynolds,
    'bounds': [(-100, 100)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 1000,
}

chung_reynolds_5000d_config = {
    'objective': chung_reynolds,
    'bounds': [(-100, 100)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 5000,
}

cosine_mixture_config = {
    'objective': cosine_mixture,
    'bounds': [(-1, 1)],
    'max_iterations': 1000,
    'global_minimum': 0.4,
    'dimensions': 4,
}

csendes_config = {
    'objective': csendes,
    'bounds': [(-1, 1), (-1, 1)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 2,
}

csendes_10d_config = {
    'objective': csendes,
    'bounds': [(-1, 1)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 10,
}

csendes_50d_config = {
    'objective': csendes,
    'bounds': [(-1, 1)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 50,
}

csendes_100d_config = {
    'objective': csendes,
    'bounds': [(-1, 1)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 100,
}

csendes_500d_config = {
    'objective': csendes,
    'bounds': [(-1, 1)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 500,
}

csendes_1000d_config = {
    'objective': csendes,
    'bounds': [(-1, 1)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 1000,
}

csendes_5000d_config = {
    'objective': csendes,
    'bounds': [(-1, 1)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 5000,
}

cube_config = {
    'objective': cube,
    'bounds': [
        (-10, 10),
        (-10, 10),
    ],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 2,
}

deb1_config = {
    'objective': deb1,
    'bounds': [(-1, 1)],
    'max_iterations': 1000,
    'global_minimum': None,
    'dimensions': 2,
}

deb1_10d_config = {
    'objective': deb1,
    'bounds': [(-1, 1)],
    'max_iterations': 1000,
    'global_minimum': None,
    'dimensions': 10,
}

deb1_50d_config = {
    'objective': deb1,
    'bounds': [(-1, 1)],
    'max_iterations': 1000,
    'global_minimum': None,
    'dimensions': 50,
}

deb1_100d_config = {
    'objective': deb1,
    'bounds': [(-1, 1)],
    'max_iterations': 1000,
    'global_minimum': None,
    'dimensions': 100,
}

deb1_500d_config = {
    'objective': deb1,
    'bounds': [(-1, 1)],
    'max_iterations': 1000,
    'global_minimum': None,
    'dimensions': 500,
}

deb1_1000d_config = {
    'objective': deb1,
    'bounds': [(-1, 1)],
    'max_iterations': 1000,
    'global_minimum': None,
    'dimensions': 1000,
}

deb1_5000d_config = {
    'objective': deb1,
    'bounds': [(-1, 1)],
    'max_iterations': 1000,
    'global_minimum': None,
    'dimensions': 5000,
}

deb3_config = {
    'objective': deb3,
    'bounds': [(-1, 1)],
    'max_iterations': 1000,
    'global_minimum': None,
    'dimensions': 2,
}

deb3_10d_config = {
    'objective': deb3,
    'bounds': [(-1, 1)],
    'max_iterations': 1000,
    'global_minimum': None,
    'dimensions': 10,
}

deb3_50d_config = {
    'objective': deb3,
    'bounds': [(-1, 1)],
    'max_iterations': 1000,
    'global_minimum': None,
    'dimensions': 50,
}

deb3_100d_config = {
    'objective': deb3,
    'bounds': [(-1, 1)],
    'max_iterations': 1000,
    'global_minimum': None,
    'dimensions': 100,
}

deb3_500d_config = {
    'objective': deb3,
    'bounds': [(-1, 1)],
    'max_iterations': 1000,
    'global_minimum': None,
    'dimensions': 500,
}

deb3_1000d_config = {
    'objective': deb3,
    'bounds': [(-1, 1)],
    'max_iterations': 1000,
    'global_minimum': None,
    'dimensions': 1000,
}

deb3_5000d_config = {
    'objective': deb3,
    'bounds': [(-1, 1)],
    'max_iterations': 1000,
    'global_minimum': None,
    'dimensions': 5000,
}

deckkers_aarts_config = {
    'objective': deckkers_aarts,
    'bounds': [
        (-20, 20),
        (-20, 20),
    ],
    'max_iterations': 1000,
    'global_minimum': -24777,
    'dimensions': 2,
}

devilliers_glasser1_config = {
    'objective': devilliers_glasser1,
    'bounds': [
        (-500, 500),
        (-500, 500),
        (-500, 500),
        (-500, 500),
    ],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 4,
}

devilliers_glasser2_config = {
    'objective': devilliers_glasser2,
    'bounds': [
        (-500, 500),
        (-500, 500),
        (-500, 500),
        (-500, 500),
        (-500, 500),
    ],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 5,
}

dixon_price_config = {
    'objective': dixon_price,
    'bounds': [(-10, 10)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 2,
}

dixon_price_10d_config = {
    'objective': dixon_price,
    'bounds': [(-10, 10)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 10,
}

dixon_price_50d_config = {
    'objective': dixon_price,
    'bounds': [(-10, 10)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 50,
}

dixon_price_100d_config = {
    'objective': dixon_price,
    'bounds': [(-10, 10)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 100,
}

dixon_price_500d_config = {
    'objective': dixon_price,
    'bounds': [(-10, 10)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 500,
}

dixon_price_1000d_config = {
    'objective': dixon_price,
    'bounds': [(-10, 10)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 1000,
}

dixon_price_5000d_config = {
    'objective': dixon_price,
    'bounds': [(-10, 10)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 5000,
}

dolan_config = {
    'objective': dolan,
    'bounds': [
        (-100, 100),
        (-100, 100),
        (-100, 100),
        (-100, 100),
        (-100, 100),
    ],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 5,
}

easom_config = {
    'objective': easom,
    'bounds': [
        (-100, 100),
        (-100, 100),
    ],
    'max_iterations': 1000,
    'global_minimum': -1,
    'dimensions': 2,
}

el_attar_config = {
    'objective': el_attar,
    'bounds': [
        (-500, 500),
        (-500, 500),
    ],
    'max_iterations': 1000,
    'global_minimum': 0.470427,
    'dimensions': 2,
}

egg_crate_config = {
    'objective': egg_crate,
    'bounds': [
        (-5, 5),
        (-5, 5),
    ],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 2,
}

exp1_config = {
    'objective': exp1,
    'bounds': [(-1, 1)],
    'max_iterations': 1000,
    'global_minimum': 1,
    'dimensions': 2,
}

exp1_10d_config = {
    'objective': exp1,
    'bounds': [(-1, 1)],
    'max_iterations': 1000,
    'global_minimum': 1,
    'dimensions': 10,
}

exp1_50d_config = {
    'objective': exp1,
    'bounds': [(-1, 1)],
    'max_iterations': 1000,
    'global_minimum': 1,
    'dimensions': 50,
}

exp1_100d_config = {
    'objective': exp1,
    'bounds': [(-1, 1)],
    'max_iterations': 1000,
    'global_minimum': 1,
    'dimensions': 100,
}

exp1_500d_config = {
    'objective': exp1,
    'bounds': [(-1, 1)],
    'max_iterations': 1000,
    'global_minimum': 1,
    'dimensions': 500,
}

exp1_1000d_config = {
    'objective': exp1,
    'bounds': [(-1, 1)],
    'max_iterations': 1000,
    'global_minimum': 1,
    'dimensions': 1000,
}

exp1_5000d_config = {
    'objective': exp1,
    'bounds': [(-1, 1)],
    'max_iterations': 1000,
    'global_minimum': 1,
    'dimensions': 5000,
}

exp2_config = {
    'objective': exp2,
    'bounds': [
        (0, 20),
        (0, 20),
    ],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 2,
}

freudenstein_roth_config = {
    'objective': freudenstein_roth,
    'bounds': [
        (-10, 10),
        (-10, 10),
    ],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 2,
}

giunta_config = {
    'objective': giunta,
    'bounds': [
        (-1, 1),
        (-1, 1),
    ],
    'max_iterations': 1000,
    'global_minimum': 0.060447,
    'dimensions': 2,
}

goldstein_price_config = {
    'objective': goldstein_price,
    'bounds': [
        (-2, 2),
        (-2, 2),
    ],
    'max_iterations': 1000,
    'global_minimum': 3,
    'dimensions': 2,
}

helical_valley_config = {
    'objective': helical_valley,
    'bounds': [(-10, 10), (-10, 10), (-10, 10)],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 3,
}

himmelblau_config = {
    'objective': himmelblau,
    'bounds': [
        (-5, 5),
        (-5, 5),
    ],
    'max_iterations': 1000,
    'global_minimum': 0,
    'dimensions': 2,
}

hosaki_config = {
    'objective': hosaki,
    'bounds': [
        (0, 5),
        (0, 6),
    ],
    'max_iterations': 1000,
    'global_minimum': -2.3458,
    'dimensions': 2,
}

jennrich_sampson_config = {
    'objective': jennrich_sampson,
    'bounds': [
        (-1, 1),
        (-1, 1),
    ],
    'max_iterations': 1000,
    'global_minimum': 124.3612,
    'dimensions': 2,
}

PROBLEMS_BY_NAME = {
    'ackley': ackley_config,
    'ackley_3d': ackley_3d_config,
    'ackley_5d': ackley_5d_config,
    'ackley_30d': ackley_30d_config,
    'ackley_100d': ackley_100d_config,
    'ackley_1000d': ackley4_1000d_config,
    'bukin_n6': bukin_n6_config,
    'cross_in_tray': cross_in_tray_config,
    'drop_wave': drop_wave_config,
    'eggholder': eggholder_config,
    'griewank': griewank_config,
    'griewank_3d': griewank_3d_config,
    'griewank_5d': griewank_5d_config,
    'griewank_30d': griewank_30d_config,
    'griewank_100d': griewank_1000d_config,
    'griewank_1000d': griewank_1000d_config,
    'holder_table': holder_table_config,
    'langermann': langermann_config,
    'levy': levy_config,
    'levy_3d': levy_3d_config,
    'levy_5d': levy_5d_config,
    'levy_30d': levy_30d_config,
    'levy_n13': levy_n13_config,
    'rastrigin': rastrigin_config,
    'rastrigin_3d': rastrigin_3d_config,
    'rastrigin_5d': rastrigin_3d_config,
    'rastrigin_30d': rastrigin_30d_config,
    'schaffer_n2': schaffer_n2_config,
    'schaffer_n4': schaffer_n4_config,
    'schwefel': schwefel_config,
    'schwefel_3d': schwefel_3d_config,
    'schwefel_5d': schwefel_5d_config,
    'schwefel_30d': schwefel_30d_config,
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
    'damavandi': damavandi_config,
    'cross_leg_table': cross_leg_table_config,
    'sine_envelope': sine_envelope_config,
    'ackley2': ackley2_config,
    'ackley3': ackley3_config,
    'ackley4': ackley4_config,
    'ackley4_10d': ackley4_10d_config,
    'ackley4_100d': ackley4_100d_config,
    'ackley4_500d': ackley4_500d_config,
    'ackley4_1000d': ackley4_1000d_config,
    'ackley4_5000d': ackley4_5000d_config,
    'adjiman': adjiman_config,
    'alpine1': alpine1_config,
    'alpine1_10d': alpine1_10d_config,
    'alpine1_100d': alpine1_100d_config,
    'alpine1_500d': alpine1_500d_config,
    'alpine1_1000d': alpine1_1000d_config,
    'alpine1_5000d': alpine1_5000d_config,
    'alpine2': alpine2_config,
    'alpine2_10d': alpine2_10d_config,
    'brad': brad_config,
    'bartels_conn': bartels_conn_config,
    'beale': beale_config,
    'biggs_exp2': biggs_exp2_config,
    'biggs_exp3': biggs_exp3_config,
    'biggs_exp4': biggs_exp4_config,
    'biggs_exp5': biggs_exp5_config,
    'biggs_exp6': biggs_exp6_config,
    'bird': bird_config,
    'bohachevsky1': bohachevsky1_config,
    'bohachevsky2': bohachevsky2_config,
    'bohachevsky3': bohachevsky3_config,
    'booth': booth_config,
    'branin_rcos': branin_rcos_config,
    'brent': brent_config,
    'brown': brown_config,
    'brown_10d': brown_10d_config,
    'brown_50d': brown_50d_config,
    'brown_100d': brown_100d_config,
    'brown_500d': brown_500d_config,
    'brown_1000d': brown_1000d_config,
    'brown_5000d': brown_5000d_config,
    'bukin_n2': bukin_n2_config,
    'bukin_n4': bukin_n4_config,
    'camel_3hump': camel_3hump_config,
    'camel_6hump': camel_6hump_config,
    'chen_bird': chen_bird_config,
    'chen_v': chen_v_config,
    'chichinadze': chichinadze_config,
    'chung_reynolds': chung_reynolds_config,
    'chung_reyonlds_10d': chung_reynolds_10d_config,
    'chung_reynolds_50d': chung_reynolds_50d_config,
    'chung_reynolds_100d': chung_reynolds_100d_config,
    'chung_reynolds_500d': chung_reynolds_500d_config,
    'chung_reynolds_1000d': chung_reynolds_1000d_config,
    'chung_reynolds_5000d': chung_reynolds_5000d_config,
    'cosine_mixture': cosine_mixture_config,
    'csendes': csendes_config,
    'csendes_10d': csendes_10d_config,
    'csendes_50d': csendes_50d_config,
    'csendes_100d': csendes_100d_config,
    'csendes_500d': csendes_500d_config,
    'csendes_1000d': csendes_1000d_config,
    'csendes_5000d': csendes_5000d_config,
    'cube': cube_config,
    'deb1': deb1_config,
    'deb1_10d': deb1_10d_config,
    'deb1_50d': deb1_50d_config,
    'deb1_100d': deb1_100d_config,
    'deb1_500d': deb1_500d_config,
    'deb1_1000d': deb1_1000d_config,
    'deb1_5000d': deb1_5000d_config,
    'deb3': deb3_config,
    'deb3_10d': deb3_10d_config,
    'deb3_50d': deb3_50d_config,
    'deb3_100d': deb3_100d_config,
    'deb3_500d': deb3_500d_config,
    'deb3_1000d': deb3_1000d_config,
    'deb3_5000d': deb3_5000d_config,
    'deckkers_aarts': deckkers_aarts_config,
    'devilliers_glasser1': devilliers_glasser1_config,
    'devilliers_glasser2': devilliers_glasser2_config,
    'dixon_price': dixon_price_config,
    'dixon_price_10d': dixon_price_10d_config,
    'dixon_price_50d': dixon_price_50d_config,
    'dixon_price_100d': dixon_price_100d_config,
    'dixon_price_500d': dixon_price_500d_config,
    'dixon_price_1000d': dixon_price_1000d_config,
    'dixon_price_5000d': dixon_price_5000d_config,
    'dolan': dolan_config,
    'easom': easom_config,
    'el_attar': el_attar_config,
    'egg_crate': egg_crate_config,
    'exp1': exp1_config,
    'exp1_10d': exp1_10d_config,
    'exp1_50d': exp1_50d_config,
    'exp1_100d': exp1_100d_config,
    'exp1_500d': exp1_500d_config,
    'exp1_1000d': exp1_1000d_config,
    'exp1_5000d': exp1_5000d_config,
    'exp2': exp2_config,
    'freudenstein_roth': freudenstein_roth_config,
    'giunta': giunta_config,
    'goldstein_price': goldstein_price_config,
    'helical_valley': helical_valley_config,
    'himmelblau': himmelblau_config,
    'hosaki': hosaki_config,
    'jennrich_sampson': jennrich_sampson_config,
}
