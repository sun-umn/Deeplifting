# stdlib
from typing import Any, Dict

# third party
import numpy as np
import pyomo.environ as pyo
import torch


# File for 2D Schaffer N2 test function
class SchafferN2:
    """
    Function that implements the Schaffer N2 function in
    numpy, pytorch or pyomo interface. We will use this
    for our deeplifting experiments.

    Schaffer N2 has a global minimum @ (0, 0) with
    f(x) = 0
    """

    def objective(self, x, version='numpy') -> float:
        """
        Schaffer N2 method
        """
        x1, x2 = x.flatten()
        if version == 'numpy':
            result = (
                0.5
                + (np.sin(x1**2 - x2**2) ** 2 - 0.5)
                / (1 + 0.001 * (x1**2 + x2**2)) ** 2
            )
        elif version == 'pyomo':
            result = (
                0.5
                + (pyo.sin(x1**2 - x2**2) ** 2 - 0.5)
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
                'Unknown version specified.'
                'Available options are numpy, pyomo and pytorch'
            )

        return result

    def config(self) -> Dict[str, Any]:
        """
        Configuration to run Schaffer N2 problem
        """
        config = {
            'objective': self.objective,
            'bounds': {
                'lower_bounds': [-100.0, -100.0],
                'upper_bounds': [100.0, 100.0],
            },
            'max_iterations': 1000,
            'global_minimum': 0.0,
            'dimensions': 2,
            'global_x': np.array([0.0, 0.0]),
            'trials': 25,
            'name': 'schaffer_n2',
        }

        return config


# File for 2D Schaffer N4 test function
class SchafferN4:
    """
    Function that implements the Schaffer N4 function in
    numpy, pytorch or pyomo interface. We will use this
    for our deeplifting experiments.

    Schaffer N4 has a global minimum @ (0.0, 1.253115)
    f(x) = 0.292579
    """

    def objective(self, x, version='numpy') -> float:
        """
        Schaffer N4 method
        """
        x1, x2 = x.flatten()
        if version == 'numpy':
            result = (
                0.5
                + (np.cos(np.sin(np.abs(x1**2 - x2**2))) ** 2 - 0.5)
                / (1 + 0.001 * (x1**2 + x2**2)) ** 2
            )
        elif version == 'pyomo':
            result = (
                0.5
                + (pyo.cos(pyo.sin(np.abs(x1**2 - x2**2))) ** 2 - 0.5)
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
                'Unknown version specified.'
                'Available options are numpy, pyomo and pytorch'
            )

        return result

    def config(self) -> Dict[str, Any]:
        """
        Configuration to run Schaffer N4 problem
        """
        config = {
            'objective': self.objective,
            'bounds': {
                'lower_bounds': [-100.0, -100.0],
                'upper_bounds': [100.0, 100.0],
            },
            'max_iterations': 1000,
            'global_minimum': 0.292579,
            'dimensions': 2,
            'global_x': np.array([0.0, 1.253115]),
            'trials': 25,
            'name': 'schaffer_n4',
        }

        return config
