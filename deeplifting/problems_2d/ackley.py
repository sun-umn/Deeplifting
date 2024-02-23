# third party
import numpy as np
import pyomo.environ as pyo
import torch


# File for 2D Ackley test function
class Ackley:
    """
    Function that implements the Ackley function in
    numpy, pytorch or pyomo interface. We will use this
    for our deeplifting experiments.

    Ackley has a global minimum @ (0, 0) with
    f(x) = 0
    """

    def ackley(self, x, version='numpy'):
        """
        Ackley method
        """
        a = 20
        b = 0.2
        c = 2 * np.pi

        # Get x1 & x2
        x1, x2 = x.flatten()

        if version == 'numpy':
            sum_sq_term = -a * np.exp(-b * np.sqrt(0.5 * ((x1) ** 2 + (x2) ** 2)))
            cos_term = -np.exp(0.5 * (np.cos(c * (x1)) + np.cos(c * (x2))))
            result = sum_sq_term + cos_term + a + np.exp(1)

        elif version == 'pyomo':
            sum_sq_term = -a * pyo.exp(-b * (0.5 * (x1**2 + x2**2) ** 0.5))
            cos_term = -pyo.exp(0.5 * (pyo.cos(c * x1) + pyo.cos(c * x2)))
            result = sum_sq_term + cos_term + a + np.e

        elif version == 'pytorch':
            sum_sq_term = -a * torch.exp(-b * torch.sqrt(0.5 * ((x1) ** 2 + (x2) ** 2)))
            cos_term = -torch.exp(0.5 * (torch.cos(c * (x1)) + torch.cos(c * (x2))))
            result = sum_sq_term + cos_term + a + torch.exp(torch.tensor(1.0))

        else:
            raise ValueError(
                'Unknown version specified. Available'
                'options are numpy, pyomo and pytorch.'
            )

        return result
