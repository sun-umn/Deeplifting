# stdlib
from typing import Any, Dict

# third party
import numpy as np
import pyomo.environ as pyo
import torch


# File for ND Schwefel test function
class Schwefel:
    """
    Function that implements the Schwefel function in
    numpy, pytorch or pyomo interface. We will use this
    for our deeplifting experiments.

    Schwefel has a global minimum @ (420.9697, 420.9687) with
    f(x) = 0
    """

    def objective(self, x, version='numpy') -> float:
        """
        Schwefel method
        """
        x = x.flatten()
        d = len(x)
        if version == 'numpy':
            result = 418.982887 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))))
        elif version == 'pyomo':
            values = [value * pyo.sin(np.abs(value) ** 0.5) for value in x]
            result = 418.982887 * d - np.sum(values)
        elif version == 'pytorch':
            result = 418.982887 * d - torch.sum(x * torch.sin(torch.abs(x) ** 0.5))
        else:
            raise ValueError(
                'Unknown version specified.'
                'Available options are numpy, pyomo and pytorch'
            )

        return result

    def config(self) -> Dict[str, Any]:
        """
        Configuration to run Schwefel problem
        """
        config = {
            'objective': self.objective,
            'bounds': {
                'lower_bounds': [-500.0, -500.0],
                'upper_bounds': [500.0, 500.0],
            },
            'max_iterations': 1000,
            'global_minimum': 0.0,
            'dimensions': 2,
            'global_x': np.array([420.9687, 420.9687]),
            'trials': 25,
            'name': 'schwefel',
        }

        return config

    def config_nd(self, dimensions) -> Dict[str, Any]:
        """'
        Method to create Ackley ND problems
        """
        config = {
            'objective': self.objective,
            'bounds': {
                'lower_bounds': [-500] * dimensions,
                'upper_bounds': [500] * dimensions,
            },
            'max_iterations': 1000,
            'global_minimum': 0.0,
            'dimensions': dimensions,
            'global_x': np.array([0.0] * dimensions),
            'trials': 15,
            'name': f'rastrigin_{dimensions}d',
        }

        return config
