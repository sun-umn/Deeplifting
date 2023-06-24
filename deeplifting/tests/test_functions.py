# stdlib
import math

# third party
import torch

# first party
from deeplifting.problems import ackley, bukin_n6


def test_ackley_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Ackley function has the correct global minimum.

    The global minimum exits as point x*=(0, 0) and
    f(x*) = 0
    """
    # Test the numpy version
    x, y = 0.0, 0.0
    result = ackley(x, y, version='numpy')
    math.isclose(result, 0.0, abs_tol=1e-15)

    # Test the torch version
    x, y = torch.tensor(0.0), torch.tensor(0.0)
    torch_result = ackley(x, y, version='pytorch').numpy()
    math.isclose(torch_result, 0.0, abs_tol=1e-7)


def test_bukin_n6_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Bukin N.6 function has the correct global minimum.

    The global minimum exits as point x*=(0, 0) and
    f(x*) = 0
    """
    # Test the numpy version
    x, y = -10.0, 1.0
    result = bukin_n6(x, y, version='numpy')
    math.isclose(result, 0.0, abs_tol=1e-7)

    # Test the torch version
    x, y = torch.tensor(-10.0), torch.tensor(1.0)
    torch_result = bukin_n6(x, y, version='pytorch').numpy()
    math.isclose(torch_result, 0.0, abs_tol=1e-7)
