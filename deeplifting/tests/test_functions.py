# stdlib
import math

# third party
import numpy as np
import torch

# first party
from deeplifting.problems import (
    ackley,
    ackley_config,
    bukin_n6,
    bukin_n6_config,
    cross_in_tray,
    cross_in_tray_config,
    cross_leg_table,
    cross_leg_table_config,
    drop_wave,
    drop_wave_config,
    eggholder,
    eggholder_config,
)


def test_ackley_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Ackley function has the correct global minimum.

    The global minimum exits as point x*=(0, 0) and
    f(x*) = 0
    """
    global_minimum = ackley_config['global_minimum']

    # Test the numpy version
    x = np.array([0.0, 0.0])
    result = ackley(x, version='numpy')
    math.isclose(result, global_minimum, abs_tol=1e-15)

    # Test the pyomo version
    result = ackley(x, version='pyomo')
    math.isclose(result, global_minimum, abs_tol=1e-15)

    # Test the torch version
    x = torch.tensor([0.0, 0.0])
    torch_result = ackley(x, version='pytorch').numpy()
    math.isclose(torch_result, global_minimum, abs_tol=1e-7)


def test_bukin_n6_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Bukin N.6 function has the correct global minimum.

    The global minimum exits as point x*=(-10, 1) and
    f(x*) = 0
    """
    global_minimum = bukin_n6_config['global_minimum']

    # Test the numpy version
    x = np.array([-10.0, 1.0])
    result = bukin_n6(x, version='numpy')
    math.isclose(result, global_minimum, abs_tol=1e-7)

    # Test pyomo version
    result = bukin_n6(x, version='pyomo')
    math.isclose(result, global_minimum, abs_tol=1e-7)

    # Test the torch version
    x = torch.tensor([-10.0, 1.0])
    torch_result = bukin_n6(x, version='pytorch').numpy()
    math.isclose(torch_result, global_minimum, abs_tol=1e-7)


def test_cross_in_tray_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Cross-In-Tray function has the correct global minimum.

    The global minimum exits as point
    x*=(1.3491, -1.3491)
    x*=(1.3491, 1.3491)
    x*=(-1.3491, 1.3491)
    x*=(-1.3491, -1.3491)
    and f(x*) = -2.06261

    We will only test one due to the symmetry
    """
    global_minimum = cross_in_tray_config['global_minimum']

    # Test the numpy version
    x = np.array([-10.0, 1.0])
    result = cross_in_tray(x, version='numpy')
    math.isclose(result, global_minimum, abs_tol=1e-7)

    # Test pyomo version
    result = cross_in_tray(x, version='pyomo')
    math.isclose(result, global_minimum, abs_tol=1e-7)

    # Test the torch version
    x = torch.tensor([-10.0, 1.0])
    torch_result = cross_in_tray(x, version='pytorch').numpy()
    math.isclose(torch_result, global_minimum, abs_tol=1e-7)


def test_cross_leg_table_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Bukin N.6 function has the correct global minimum.

    The global minimum exits as point x*=(0.0, 0.0) and
    f(x*) = 0
    """
    global_minimum = cross_leg_table_config['global_minimum']

    # Test the numpy version
    x = np.array([0.0, 0.0])
    result = cross_leg_table(x, version='numpy')
    math.isclose(result, global_minimum, abs_tol=1e-7)

    # Test pyomo version
    result = cross_leg_table(x, version='pyomo')
    math.isclose(result, global_minimum, abs_tol=1e-7)

    # Test the torch version
    x = torch.tensor([0.0, 0.0])
    torch_result = cross_leg_table(x, version='pytorch').numpy()
    math.isclose(torch_result, global_minimum, abs_tol=1e-7)


def test_drop_wave_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Drop Wave function has the correct global minimum.

    The global minimum exits as point x*=(0, 0) and
    f(x*) = -1
    """
    global_minimum = drop_wave_config['global_minimum']

    # Test the numpy version
    x = np.array([0.0, 0.0])
    result = drop_wave(x, version='numpy')
    math.isclose(result, global_minimum, abs_tol=1e-7)

    result = drop_wave(x, version='pyomo')
    math.isclose(result, global_minimum, abs_tol=1e-7)

    # Test the torch version
    x = torch.tensor([0.0, 0.0])
    torch_result = drop_wave(x, version='pytorch').numpy()
    math.isclose(torch_result, global_minimum, abs_tol=1e-7)


def test_eggholder_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Eggholder function has the correct global minimum.

    The global minimum exits as point x*=(512, 404.2319) and
    f(x*) = -959.6407
    """
    global_minimum = eggholder_config['global_minimum']

    # Test the numpy version
    x = np.array([512.0, 404.2319])
    result = eggholder(x, version='numpy')
    math.isclose(result, global_minimum, abs_tol=1e-3)

    result = eggholder(x, version='pyomo')
    math.isclose(result, global_minimum, abs_tol=1e-3)

    # Test the torch version
    x = torch.tensor([512.0, 404.2319])
    torch_result = eggholder(x, version='pytorch').numpy()
    math.isclose(torch_result, global_minimum, abs_tol=1e-3)
