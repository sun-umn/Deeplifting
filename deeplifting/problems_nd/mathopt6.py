# stdlib
from typing import Any, Dict

# third party
import numpy as np
import pyomo.environ as pyo
import torch


class MathOpt6:
    """
    Function that implements the MathOpt6 function in
    numpy, pytorch or pyomo interface. We will use this
    for our deeplifting experiments.

    Levy has a global minimum @ (-0.024399, 0.210612) with
    f(x) = -3.3069,
    """

    def objective(self, x, version='numpy') -> float:
        """
        MathOpt6 method
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
        elif version == 'pyomo':
            result = (
                pyo.exp(pyo.sin(50 * x1))
                + pyo.sin(60 * pyo.exp(x2))
                + pyo.sin(70 * pyo.sin(x1))
                + pyo.sin(pyo.sin(80 * x2))
                - pyo.sin(10 * x1 + 10 * x2)
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
                'lower_bounds': [-3.0, -3.0],
                'upper_bounds': [3.0, 3.0],
            },
            'max_iterations': 1000,
            'global_minimum': -3.3069,
            'dimensions': 2,
            'global_x': np.array([0.0, 0.0]),
            'trials': 25,
            'name': 'mathopt6',
        }

        return config
