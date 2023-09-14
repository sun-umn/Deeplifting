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
    griewank,
    griewank_config,
    holder_table,
    holder_table_config,
    levy,
    levy_config,
    levy_n13,
    levy_n13_config,
    rastrigin,
    rastrigin_config,
    schaffer_n2,
    schaffer_n2_config,
    schaffer_n4,
    schaffer_n4_config,
    schwefel,
    schwefel_config,
    shubert,
    shubert_config,
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
    assert math.isclose(result, global_minimum, abs_tol=1e-7)

    # Test the pyomo version
    result = ackley(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-7)

    # Test the torch version
    x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    torch_result = ackley(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-7)


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
    assert math.isclose(result, global_minimum, abs_tol=1e-5)

    # Test pyomo version
    result = bukin_n6(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-5)

    # Test the torch version
    x = torch.tensor([-10.0, 1.0], dtype=torch.float64)
    torch_result = bukin_n6(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-5)


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
    x = np.array([1.3491, 1.3491])
    result = cross_in_tray(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-5)

    # Test pyomo version
    result = cross_in_tray(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-5)

    # Test the torch version
    x = torch.tensor([-1.3491, 1.3491], dtype=torch.float64)
    torch_result = cross_in_tray(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-5)


def test_cross_leg_table_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Cross Leg Table function has the correct global minimum.

    The global minimum exits as point x*=(0.0, 0.0) and
    f(x*) = 0
    """
    global_minimum = cross_leg_table_config['global_minimum']

    # Test the numpy version
    x = np.array([0.0, 0.0])
    result = cross_leg_table(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-5)

    # Test pyomo version
    result = cross_leg_table(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-5)

    # Test the torch version
    x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    torch_result = cross_leg_table(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-5)


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
    assert math.isclose(result, global_minimum, abs_tol=1e-5)

    result = drop_wave(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-5)

    # Test the torch version
    x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    torch_result = drop_wave(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-5)


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
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    result = eggholder(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    # Test the torch version
    x = torch.tensor([512.0, 404.2319], dtype=torch.float64)
    torch_result = eggholder(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-4)


def test_griewank_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Griewank function has the correct global minimum.

    The global minimum exits as point x*=(0.0, 0.0) and
    f(x*) = 0.0
    """
    global_minimum = griewank_config['global_minimum']

    # Test the numpy version
    x = np.array([0.0, 0.0])
    result = griewank(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-5)

    result = griewank(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-5)

    # Test the torch version
    x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    torch_result = griewank(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-5)


def test_holder_table_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Holder Table function has the correct global minimum.

    The global minimum exits
    x*=(8.05502, 9.66459)
    x*=(-8.05502, 9.66459)
    x*=(8.05502, -9.66459)
    x*=(-8.05502, -9.66459)
    and f(x*) = -19.2085
    """
    global_minimum = holder_table_config['global_minimum']

    # Test the numpy version
    x = np.array([-8.05502, -9.66459])
    result = holder_table(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-5)

    result = holder_table(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-5)

    # Test the torch version
    x = torch.tensor([8.05502, -9.66459], dtype=torch.float64)
    torch_result = holder_table(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-5)


def test_levy_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Levy function has the correct global minimum.

    The global minimum exits
    x*=(1.0, 1.0)
    f(x*) = 0
    """
    global_minimum = levy_config['global_minimum']

    # Test the numpy version
    x = np.array([1.0, 1.0])
    result = levy(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    result = levy(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    # Test the torch version
    x = torch.tensor([1.0, 1.0], dtype=torch.float64)
    torch_result = levy(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-4)


def test_levy_n13_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Levy N13 function has the correct global minimum.

    The global minimum exits
    x*=(1.0, 1.0)
    f(x*) = 0
    """
    global_minimum = levy_n13_config['global_minimum']

    # Test the numpy version
    x = np.array([1.0, 1.0])
    result = levy_n13(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    result = levy_n13(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    # Test the torch version
    x = torch.tensor([1.0, 1.0], dtype=torch.float64)
    torch_result = levy_n13(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-4)


def test_rastrigin_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Rastrigin function has the correct global minimum.

    The global minimum exits
    x*=(0.0, 0.0)
    f(x*) = 0
    """
    global_minimum = rastrigin_config['global_minimum']

    # Test the numpy version
    x = np.array([0.0, 0.0])
    result = rastrigin(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    result = rastrigin(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    # Test the torch version
    x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    torch_result = rastrigin(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-4)


def test_schaffer_n2_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Schaffer N2 function has the correct global minimum.

    The global minimum exits
    x*=(0.0, 0.0)
    f(x*) = 0
    """
    global_minimum = schaffer_n2_config['global_minimum']

    # Test the numpy version
    x = np.array([0.0, 0.0])
    result = schaffer_n2(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    result = schaffer_n2(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    # Test the torch version
    x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    torch_result = schaffer_n2(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-4)


def test_schaffer_n4_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Schaffer N4 function has the correct global minimum.

    The global minimum exits
    x*=(0.0, 1.253115)
    f(x*) = 0.292579
    """
    global_minimum = schaffer_n4_config['global_minimum']

    # Test the numpy version
    x = np.array([0.0, 1.253115])
    result = schaffer_n4(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    result = schaffer_n4(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    # Test the torch version
    x = torch.tensor([0.0, 1.253115], dtype=torch.float64)
    torch_result = schaffer_n4(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-4)


def test_schwefel_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Schwefel function has the correct global minimum.

    The global minimum exits
    x*=(420.9687, 420.9687)
    f(x*) = 0.0
    """
    global_minimum = schwefel_config['global_minimum']

    # Test the numpy version
    x = np.array([420.9687, 420.9687])
    result = schwefel(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    result = schwefel(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    # Test the torch version
    x = torch.tensor([420.9687, 420.9687], dtype=torch.float64)
    torch_result = schwefel(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-4)


def test_schubert_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Shubert function has the correct global minimum.

    The function has many global minimum values: Here is
    one example
    x*=(-7.0835, 4.8580)
    x*=(5.4828, 4.8580)
    f(x*) = -186.7309
    """
    global_minimum = shubert_config['global_minimum']

    # Test the numpy version
    x = np.array([-7.083506, -1.425128])
    result = shubert(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    result = shubert(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    # Test the torch version
    x = torch.tensor([-7.083506, -1.425128], dtype=torch.float64)
    torch_result = shubert(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-4)
