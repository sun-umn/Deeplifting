# stdlib
import math

# third party
import numpy as np
import torch

# first party
from deeplifting.problems import ackley, bukin_n6, drop_wave, eggholder


def test_ackley_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Ackley function has the correct global minimum.

    The global minimum exits as point x*=(0, 0) and
    f(x*) = 0
    """
    # Test the numpy version
    x = np.array([0.0, 0.0])
    result = ackley(x, version='numpy')
    math.isclose(result, 0.0, abs_tol=1e-15)

    # Test the torch version
    x = torch.tensor([0.0, 0.0])
    torch_result = ackley(x, version='pytorch').numpy()
    math.isclose(torch_result, 0.0, abs_tol=1e-7)


def test_bukin_n6_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Bukin N.6 function has the correct global minimum.

    The global minimum exits as point x*=(-10, 1) and
    f(x*) = 0
    """
    # Test the numpy version
    x = np.array([-10.0, 1.0])
    result = bukin_n6(x, version='numpy')
    math.isclose(result, 0.0, abs_tol=1e-7)

    # Test the torch version
    x = torch.tensor([-10.0, 1.0])
    torch_result = bukin_n6(x, version='pytorch').numpy()
    math.isclose(torch_result, 0.0, abs_tol=1e-7)


def test_drop_wave_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Drop Wave function has the correct global minimum.

    The global minimum exits as point x*=(0, 0) and
    f(x*) = -1
    """
    # Test the numpy version
    x = np.array([0.0, 0.0])
    result = drop_wave(x, version='numpy')
    math.isclose(result, -1.0, abs_tol=1e-7)

    # Test the torch version
    x = torch.tensor([0.0, 0.0])
    torch_result = drop_wave(x, version='pytorch').numpy()
    math.isclose(torch_result, -1.0, abs_tol=1e-7)


def test_eggholder_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Eggholder function has the correct global minimum.

    The global minimum exits as point x*=(512, 404.2319) and
    f(x*) = -959.6407
    """
    # Test the numpy version
    x = np.array([512.0, 404.2319])
    result = eggholder(x, version='numpy')
    math.isclose(result, -959.6407, abs_tol=1e-3)

    # Test the torch version
    x = torch.tensor([512.0, 404.2319])
    torch_result = eggholder(x, version='pytorch').numpy()
    math.isclose(torch_result, -959.6407, abs_tol=1e-3)
