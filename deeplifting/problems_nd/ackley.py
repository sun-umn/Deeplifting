# stdlib
from typing import Any, Dict

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

    def objective(self, x, version='numpy') -> float:
        """
        Ackley method
        """
        a = 20
        b = 0.2
        c = 2 * np.pi

        d = len(x)
        x = x.flatten()

        if version == 'numpy':
            arg1 = -b * (1.0 / d * np.sum(np.square(x))) ** 0.5
            arg2 = 1.0 / d * np.sum(np.cos(c * x))
            result = -a * np.exp(arg1) - np.exp(arg2) + a + np.e
        elif version == 'pyomo':
            arg1 = -b * (1.0 / d * np.sum(x**2)) ** 0.5
            values = [pyo.cos(c * value) for value in x]
            arg2 = 1.0 / d * np.sum(values)
            result = -a * pyo.exp(arg1) - pyo.exp(arg2) + a + np.e
        elif version == 'pytorch':
            arg1 = -b * torch.sqrt(1.0 / d * torch.sum(x**2))
            arg2 = 1.0 / d * torch.sum(torch.cos(c * x))
            result = -a * torch.exp(arg1) - torch.exp(arg2) + a + np.e
        else:
            raise ValueError(
                'Unknown version specified.'
                'Available options are numpy, pyomo and pytorch'
            )

        return result

    def config(self) -> Dict[str, Any]:
        """
        Configuration to run Ackley problem
        """
        config = {
            'objective': self.objective,
            'bounds': {
                'lower_bounds': [-32.768, -32.768],
                'upper_bounds': [32.768, 32.768],
            },
            'max_iterations': 1000,
            'global_minimum': 0.0,
            'dimensions': 2,
            'global_x': np.array([0.0, 0.0]),
            'trials': 25,
            'name': 'ackley',
        }

        return config

    def config_nd(self, dimensions) -> Dict[str, Any]:
        """'
        Method to create Ackley ND problems
        """
        config = {
            'objective': self.objective,
            'bounds': {
                'lower_bounds': [-32.768] * dimensions,
                'upper_bounds': [32.768] * dimensions,
            },
            'max_iterations': 1000,
            'global_minimum': 0.0,
            'dimensions': 2,
            'global_x': np.array([0.0] * dimensions),
            'trials': 15,
            'name': f'ackley_{dimensions}d',
        }

        return config
