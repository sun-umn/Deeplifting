# stdlib
from typing import Any, Dict

# third party
import numpy as np
import pyomo.environ as pyo
import torch


# File for ND Rastrigin test function
class Rastrigin:
    """
    Function that implements the Rastrigin function in
    numpy, pytorch or pyomo interface. We will use this
    for our deeplifting experiments.

    Rastrigin has a global minimum @ (0, 0) with
    f(x) = 0
    """

    def objective(self, x, version='numpy') -> float:
        """
        Rastrigin method
        """
        x = x.flatten()
        d = len(x)
        if version == 'numpy':
            result = 10 * d + np.sum(np.square(x) - 10 * np.cos(2 * np.pi * x))
        elif version == 'pyomo':
            values = [value**2 - 10 * pyo.cos(2.0 * np.pi * value) for value in x]
            result = 10 * d + np.sum(values)
        elif version == 'pytorch':
            result = 10 * d + torch.sum(torch.square(x) - 10 * torch.cos(2 * np.pi * x))
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
                'lower_bounds': [-5.12, -5.12],
                'upper_bounds': [5.12, 5.12],
            },
            'max_iterations': 1000,
            'global_minimum': 0.0,
            'dimensions': 2,
            'global_x': np.array([0.0, 0.0]),
            'trials': 25,
            'name': 'rastrigin',
        }

        return config

    def config_nd(self, dimensions) -> Dict[str, Any]:
        """'
        Method to create Ackley ND problems
        """
        config = {
            'objective': self.objective,
            'bounds': {
                'lower_bounds': [-5.12] * dimensions,
                'upper_bounds': [5.12] * dimensions,
            },
            'max_iterations': 1000,
            'global_minimum': 0.0,
            'dimensions': 2,
            'global_x': np.array([0.0] * dimensions),
            'trials': 15,
            'name': f'rastrigin_{dimensions}d',
        }

        return config
