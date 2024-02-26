# stdlib
from typing import Any, Dict

# third party
import numpy as np
import pyomo.environ as pyo
import torch


# File for 2D Levy test function
class Levy:
    """
    Function that implements the Levy function in
    numpy, pytorch or pyomo interface. We will use this
    for our deeplifting experiments.

    Levy has a global minimum @ (1, 1) with
    f(x) = 0
    """

    def objective(self, x, version='numpy') -> float:
        """
        Levy method
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

        elif version == 'pyomo':
            w1 = 1 + (x1 - 1) / 4
            w2 = 1 + (x2 - 1) / 4
            result = (
                pyo.sin(np.pi * w1) ** 2
                + (w2 - 1) ** 2 * (1 + 10 * pyo.sin(np.pi * w2 + 1) ** 2)
                + (w1 - 1) ** 2 * (1 + pyo.sin(2 * np.pi * w1) ** 2)
            )

        elif version == 'pytorch':
            w1 = 1 + (x1 - 1) / 4
            w2 = 1 + (x2 - 1) / 4
            result = (
                torch.sin(torch.tensor(np.pi) * w1) ** 2
                + (w2 - 1) ** 2
                * (1 + 10 * torch.sin(torch.tensor(np.pi) * w2 + 1) ** 2)
                + (w1 - 1) ** 2 * (1 + torch.sin(2 * torch.tensor(np.pi) * w1) ** 2)
            )

        else:
            raise ValueError(
                'Unknown version specified. Available options are numpy and pytorch.'
            )

        return result

    def config(self) -> Dict[str, Any]:
        """
        Configuration to run Ackley problem
        """
        config = {
            'objective': self.objective,
            'bounds': {
                'lower_bounds': [-10.0, -10.0],
                'upper_bounds': [10.0, 10.0],
            },
            'max_iterations': 1000,
            'global_minimum': 0.0,
            'dimensions': 2,
            'global_x': np.array([1.0, 1.0]),
            'trials': 25,
            'name': 'levy',
        }

        return config
