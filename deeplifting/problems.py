# third party
import numpy as np
import torch


def ackley(x, y, version='numpy'):
    """
    Function that implements the Ackley function in
    numpy or pytorch. We will use this for our deeplifting experiments.
    Note, that this version is the 2-D version only.
    """
    a = 20
    b = 0.2
    c = 2 * np.pi

    if version == 'numpy':
        sum_sq_term = -a * np.exp(-b * np.sqrt(0.5 * (x**2 + y**2)))
        cos_term = -np.exp(0.5 * (np.cos(c * x) + np.cos(c * y)))
        result = sum_sq_term + cos_term + a + np.exp(1)
    elif version == 'pytorch':
        sum_sq_term = -a * torch.exp(-b * torch.sqrt(0.5 * (x**2 + y**2)))
        cos_term = -torch.exp(0.5 * (torch.cos(c * x) + torch.cos(c * y)))
        result = sum_sq_term + cos_term + a + torch.exp(torch.tensor(1.0))
    else:
        raise ValueError(
            "Unknown version specified. Available " "options are 'numpy' and 'pytorch'."
        )

    return result


def bukin_n6(x, y, version='numpy'):
    """
    Function that implements the Bukin Function N.6 in both
    numpy and pytorch.
    """
    if version == 'numpy':
        term1 = 100 * np.sqrt(np.abs(y - 0.01 * x**2))
        term2 = 0.01 * np.abs(x + 10)
        result = term1 + term2
    elif version == 'pytorch':
        term1 = 100 * torch.sqrt(torch.abs(y - 0.01 * x**2))
        term2 = 0.01 * torch.abs(x + 10)
        result = term1 + term2
    else:
        raise ValueError(
            "Unknown version specified. Available options are 'numpy' and 'pytorch'."
        )

    return result
