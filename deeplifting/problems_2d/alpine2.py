# stdlib
from typing import Any, Dict

# third party
import numpy as np
import pyomo.environ as pyo
import torch


# File for 2D Alpine2 test function
class Alpine2:
    """
    Function that implements the Ackley function in
    numpy, pytorch or pyomo interface. We will use this
    for our deeplifting experiments.

    This is a 2-dimensional function with a global minimum of 2.808^2
    at (7.917,7.917)
    """

    def objective(self, x, version='numpy'):
        """
        Implementation of the Alpine2 function.

        This is the correct version:
        https://towardsdatascience.com/optimization-eye-pleasure-78-benchmark-test-functions-for-single-objective-optimization-92e7ed1d1f12  # noqa
        """
        x1, x2 = x.flatten()
        if version == 'numpy':
            result = -1.0 * (np.sqrt(x1) * np.sin(x1)) * (np.sqrt(x2) * np.sin(x2))

        elif version == 'pyomo':
            result = -1.0 * (x1**0.5 * pyo.sin(x1)) * (x2**0.5 * pyo.sin(x2))

        elif version == 'pytorch':
            result = (
                -1.0
                * (torch.sqrt(x1) * torch.sin(x1))
                * (torch.sqrt(x2) * torch.sin(x2))
            )

        else:
            raise ValueError(
                'Unknown version specified. Available'
                'options are numpy, pyomo and pytorch.'
            )

        return result

    def config(self) -> Dict[str, Any]:
        """
        Configuration to run Alpine2 problem
        """
        config = {
            'objective': self.objective,
            'bounds': {
                'lower_bounds': [0.0, 0.0],
                'upper_bounds': [10.0, 10.0],
            },
            'max_iterations': 1000,
            'global_minimum': -7.885600,
            'dimensions': 2,
            'global_x': np.array([7.917, 7.917]),
            'trials': 25,
            'name': 'alpine2',
        }

        return config
