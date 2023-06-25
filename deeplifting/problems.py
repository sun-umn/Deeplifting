# third party
import numpy as np
import torch


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
            (x1.detach().numpy(), x2.detach().numpy(), result.detach().numpy())
        )

    else:
        results[trial, iteration, :] = np.array((x1, x2, result))

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


# Problem configurations
# Ackley
ackley_config = {
    'objective': ackley,
    'bounds': [(-32.768, 32.768), (-32.768, 32.768)],
    'max_iterations': 1000,
}

# Bukin N.6
bukin_n6_config = {
    'objective': bukin_n6,
    'bounds': [(-15.0, -5.0), (-3.0, 3.0)],
    'max_iterations': 1000,
}

# Cross-in-Tray
cross_in_tray_config = {
    'objective': cross_in_tray,
    'bounds': [(-10.0, 10.0), (-10.0, 10.0)],
    'max_iterations': 1000,
}

# Drop Wave
drop_wave_config = {
    'objective': drop_wave,
    'bounds': [(-5.12, 5.12), (-5.12, 5.12)],
    'max_iterations': 1000,
}

# Eggholder
eggholder_config = {
    'objective': eggholder,
    'bounds': [(-512.0, 512.0), (-512.0, 512.0)],
    'max_iterations': 1000,
}

# Griewank
griewank_config = {
    'objective': griewank,
    'bounds': [(-600.0, 600.0), (-600.0, 600.0)],
    'max_iterations': 1000,
}

# Holder Table
holder_table_config = {
    'objective': holder_table,
    'bounds': [(-10.0, 10.0), (-10.0, 10.0)],
    'max_iterations': 1000,
}

# Levy
levy_config = {
    'objective': levy,
    'bounds': [(-10.0, 10.0), (-10.0, 10.0)],
    'max_iterations': 1000,
}

# Levy
levy_n13_config = {
    'objective': levy_n13,
    'bounds': [(-10.0, 10.0), (-10.0, 10.0)],
    'max_iterations': 1000,
}


PROBLEMS_BY_NAME = {
    'ackley': ackley_config,
    'bukin_n6': bukin_n6_config,
    'cross_in_tray': cross_in_tray_config,
    'drop_wave': drop_wave_config,
    'eggholder': eggholder_config,
    'griewank': griewank_config,
    'holder_table': holder_table_config,
    'levy': levy_config,
    'levy_n13': levy_n13_config,
}
