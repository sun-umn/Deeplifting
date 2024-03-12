# stdlib
import math

# third party
import numpy as np
import torch

# first party
from deeplifting.problems import (
    ackley2,
    ackley2_config,
    ackley3,
    ackley3_config,
    adjiman,
    adjiman_config,
    alpine1,
    alpine1_config,
    bartels_conn,
    bartels_conn_config,
    beale,
    beale_config,
    bird,
    bird_config,
    bohachevsky1,
    bohachevsky1_config,
    bohachevsky2,
    bohachevsky2_config,
    bohachevsky3,
    bohachevsky3_config,
    booth,
    booth_config,
    branin_rcos,
    branin_rcos_config,
    brent,
    brent_config,
    bukin_n2,
    bukin_n2_config,
    bukin_n4,
    bukin_n4_config,
    bukin_n6,
    bukin_n6_config,
    camel_3hump,
    camel_3hump_config,
    camel_6hump,
    camel_6hump_config,
    chung_reynolds,
    chung_reynolds_config,
    cross_in_tray,
    cross_in_tray_config,
    cross_leg_table,
    cross_leg_table_config,
    cube,
    cube_config,
    drop_wave,
    drop_wave_config,
    eggholder,
    eggholder_config,
    ex8_1_1,
    ex8_1_1_config,
    griewank,
    griewank_config,
    holder_table,
    holder_table_config,
    rosenbrock,
    rosenbrock_config,
    schaffer_n2,
    schaffer_n2_config,
    schaffer_n4,
    schaffer_n4_config,
    schwefel,
    schwefel_config,
    shubert,
    shubert_config,
    xinsheyang_n2,
    xinsheyang_n2_config,
    xinsheyang_n3,
    xinsheyang_n3_config,
)
from deeplifting.problems_nd.ackley import Ackley
from deeplifting.problems_nd.alpine2 import Alpine2
from deeplifting.problems_nd.levy import Levy, LevyN13
from deeplifting.problems_nd.mathopt6 import MathOpt6
from deeplifting.problems_nd.rastrigin import Rastrigin


def test_ackley_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Ackley function has the correct global minimum.

    The global minimum exits as point x*=(0, 0) and
    f(x*) = 0
    """
    ackley = Ackley()
    ackley_config = ackley.config()

    global_minimum = ackley_config['global_minimum']

    # Test the numpy version
    x = np.array([0.0, 0.0])
    result = ackley.objective(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-7)

    # Test the pyomo version
    result = ackley.objective(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-7)

    # Test the torch version
    x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    torch_result = ackley.objective(x, version='pytorch').numpy()
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
    levy = Levy()
    levy_config = levy.config()
    global_minimum = levy_config['global_minimum']

    # Test the numpy version
    x = np.array([1.0, 1.0])
    result = levy.objective(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    result = levy.objective(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    # Test the torch version
    x = torch.tensor([1.0, 1.0], dtype=torch.float64)
    torch_result = levy.objective(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-4)


def test_levy_n13_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Levy N13 function has the correct global minimum.

    The global minimum exits
    x*=(1.0, 1.0)
    f(x*) = 0
    """
    levy_n13 = LevyN13()
    levy_n13_config = levy_n13.config()
    global_minimum = levy_n13_config['global_minimum']

    # Test the numpy version
    x = np.array([1.0, 1.0])
    result = levy_n13.objective(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    result = levy_n13.objective(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    # Test the torch version
    x = torch.tensor([1.0, 1.0], dtype=torch.float64)
    torch_result = levy_n13.objective(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-4)


def test_rastrigin_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Rastrigin function has the correct global minimum.

    The global minimum exits
    x*=(0.0, 0.0)
    f(x*) = 0
    """
    rastrigin = Rastrigin()
    rastrigin_config = rastrigin.config()
    global_minimum = rastrigin_config['global_minimum']

    # Test the numpy version
    x = np.array([0.0, 0.0])
    result = rastrigin.objective(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    result = rastrigin.objective(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    # Test the torch version
    x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    torch_result = rastrigin.objective(x, version='pytorch').numpy()
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


def test_ex8_1_1_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Ex 8-1-1 function has the correct global minimum.
    This problem comes from the MINLP library

    The function has many global minimum values: Here is
    one example
    x*=(2.0, 0.1057835)
    f(x*) = -2.021807
    """
    global_minimum = ex8_1_1_config['global_minimum']

    # Test the numpy version
    x = np.array([2.0, 0.1057835])
    result = ex8_1_1(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    result = ex8_1_1(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    # Test the torch version
    x = torch.tensor([2.0, 0.1057835], dtype=torch.float64)
    torch_result = ex8_1_1(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-4)


def test_mathopt6_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Ex mathopt6 function has the correct global minimum.
    This problem comes from the MINLP library

    The function has many global minimum values: Here is
    one example
    x*=(-0.024399, 0.210612)
    f(x*) = -3.306869
    """
    mathopt6 = MathOpt6()
    mathopt6_config = mathopt6.config()
    global_minimum = mathopt6_config['global_minimum']

    # Test the numpy version
    x = np.array([-0.024399, 0.210612])
    result = mathopt6.objective(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    result = mathopt6.objective(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    # Test the torch version
    x = torch.tensor([-0.024399, 0.210612], dtype=torch.float64)
    torch_result = mathopt6.objective(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-4)


def test_rosenbrock_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Ex Rosenbrock function has the correct global minimum.
    This problem comes from the MINLP library

    The function has many global minimum values: Here is
    one example
    x*=(0.9999, 0.9999)
    f(x*) = 0.0
    """
    global_minimum = rosenbrock_config['global_minimum']

    # Test the numpy version
    x = np.array([0.99999, 0.99999])
    result = rosenbrock(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    result = rosenbrock(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    # Test the torch version
    x = torch.tensor([0.99999, 0.99999], dtype=torch.float64)
    torch_result = rosenbrock(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-4)


def test_ackley2_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Ex Ackley 2 function has the correct global minimum.

    The function has many global minimum values: Here is
    one example
    x*=(0.0, 0.0)
    f(x*) = -200.0
    """
    global_minimum = ackley2_config['global_minimum']

    # Test the numpy version
    x = np.array([0.0, 0.0])
    result = ackley2(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    result = ackley2(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    # Test the torch version
    x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    torch_result = ackley2(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-4)


def test_ackley3_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Ex Ackley 3 function has the correct global minimum.

    The function has many global minimum values: Here is
    one example
    x*=(0.682584587365898, -0.36075325513719)
    f(x*) = -195.629028238419
    """
    global_minimum = ackley3_config['global_minimum']

    # Test the numpy version
    x = np.array([0.682584587365898, -0.36075325513719])
    result = ackley3(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    result = ackley3(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    # Test the torch version
    x = torch.tensor([0.682584587365898, -0.36075325513719], dtype=torch.float64)
    torch_result = ackley3(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-4)


def test_adjiman_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Ex Adjiman function has the correct global minimum.

    The function has many global minimum values: Here is
    one example
    x*=(2.0, 0.1057835)
    f(x*) = -2.021807
    """
    global_minimum = adjiman_config['global_minimum']

    # Test the numpy version
    x = np.array([2.0, 0.1057835])
    result = adjiman(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    result = adjiman(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    # Test the torch version
    x = torch.tensor([2.0, 0.1057835], dtype=torch.float64)
    torch_result = adjiman(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-4)


def test_alpine1_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Ex Alpine 1 function has the correct global minimum.

    The function has many global minimum values: Here is
    one example
    x*=(0.0, 0.0)
    f(x*) = 0.0
    """
    global_minimum = alpine1_config['global_minimum']

    # Test the numpy version
    x = np.array([0.0, 0.0])
    result = alpine1(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    result = alpine1(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-4)

    # Test the torch version
    x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    torch_result = alpine1(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-4)


def test_alpine2_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Alpine 2 function has the correct global minimum.

    The function has one global minima
    x* = (7.917, 7.917)
    f(x*) = -2.808 ** 2 for 2D
    """
    alpine2 = Alpine2()
    alpine2_config = alpine2.config()
    global_minimum = alpine2_config['global_minimum']

    # Test the numpy version
    x = np.array([7.917, 7.917])
    result = alpine2.objective(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    result = alpine2.objective(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    # Test the torch version
    x = torch.tensor([7.917, 7.917], dtype=torch.float64)
    torch_result = alpine2.objective(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-2)


def test_bartels_conn_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Ex Bartels Conn function has the correct global minimum.

    The function has many global minimum values: Here is
    one example
    x*=(0.0, 0.0)
    f(x*) = 1.0
    """
    global_minimum = bartels_conn_config['global_minimum']

    # Test the numpy version
    x = np.array([0.0, 0.0])
    result = bartels_conn(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    result = bartels_conn(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    # Test the torch version
    x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    torch_result = bartels_conn(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-2)


def test_beale_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Ex Beale function has the correct global minimum.

    The function has many global minimum values: Here is
    one example
    x*=(3.0, 0.5)
    f(x*) = 0.0
    """
    global_minimum = beale_config['global_minimum']

    # Test the numpy version
    x = np.array([3.0, 0.5])
    result = beale(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    result = beale(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    # Test the torch version
    x = torch.tensor([3.0, 0.5], dtype=torch.float64)
    torch_result = beale(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-2)


def test_bird_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Ex Bird function has the correct global minimum.

    The function has many global minimum values: Here is
    one example
    x*=(4.70104, 3.15294)
    x*=(-1.58214, -3.13024)
    f(x*) = -106.764537
    """
    global_minimum = bird_config['global_minimum']

    # Test the numpy version
    x = np.array([4.70104, 3.15294])
    result = bird(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    result = bird(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    # Test the torch version
    x = torch.tensor([-1.58214, -3.13024], dtype=torch.float64)
    torch_result = bird(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-2)


def test_bohachevsky1_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Ex Bohachevsky 1 function has the correct global minimum.

    The function has many global minimum values: Here is
    one example
    x*=(0.0, 0.0)
    f(x*) = 0.0
    """
    global_minimum = bohachevsky1_config['global_minimum']

    # Test the numpy version
    x = np.array([0.0, 0.0])
    result = bohachevsky1(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    result = bohachevsky1(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    # Test the torch version
    x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    torch_result = bohachevsky1(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-2)


def test_bohachevsky2_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Ex Bohachevsky 2 function has the correct global minimum.

    The function has many global minimum values: Here is
    one example
    x*=(0.0, 0.0)
    f(x*) = 0.0
    """
    global_minimum = bohachevsky2_config['global_minimum']

    # Test the numpy version
    x = np.array([0.0, 0.0])
    result = bohachevsky2(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    result = bohachevsky2(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    # Test the torch version
    x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    torch_result = bohachevsky2(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-2)


def test_bohachevsky3_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Ex Bohachevsky 3 function has the correct global minimum.

    The function has many global minimum values: Here is
    one example
    x*=(0.0, 0.0)
    f(x*) = 0.0
    """
    global_minimum = bohachevsky3_config['global_minimum']

    # Test the numpy version
    x = np.array([0.0, 0.0])
    result = bohachevsky3(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    result = bohachevsky3(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    # Test the torch version
    x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    torch_result = bohachevsky3(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-2)


def test_booth_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Ex Booth function has the correct global minimum.

    The function has many global minimum values: Here is
    one example
    x*=(1.0, 3.0)
    f(x*) = 0.0
    """
    global_minimum = booth_config['global_minimum']

    # Test the numpy version
    x = np.array([1.0, 3.0])
    result = booth(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    result = booth(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    # Test the torch version
    x = torch.tensor([1.0, 3.0], dtype=torch.float64)
    torch_result = booth(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-2)


def test_branin_rcos_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Ex Branin RCOS function has the correct global minimum.

    The function has many global minimum values: Here is
    one example
    x*=(-pi, 12.275)
    x*=(3 * pi, 2.425)
    f(x*) = 0.3978873
    """
    global_minimum = branin_rcos_config['global_minimum']

    # Test the numpy version
    x = np.array([-np.pi, 12.275])
    result = branin_rcos(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    result = branin_rcos(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    # Test the torch version
    x = torch.tensor([3 * torch.pi, 2.425], dtype=torch.float64)
    torch_result = branin_rcos(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-2)


def test_brent_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Ex Brent function has the correct global minimum.

    The function has many global minimum values: Here is
    one example
    x*=(-10.0, -10.0)
    f(x*) = 0.0

    x* verified from here:
    https://towardsdatascience.com/optimization-eye-pleasure-78-benchmark-test-functions-for-single-objective-optimization-92e7ed1d1f12  # noqa
    """
    global_minimum = brent_config['global_minimum']

    # Test the numpy version
    x = np.array([-10.0, -10.0])
    result = brent(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    result = brent(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    # Test the torch version
    x = torch.tensor([-10.0, -10.0], dtype=torch.float64)
    torch_result = brent(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-2)


def test_bukin_n2_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Ex Bukin N2 function has the correct global minimum.

    The function has many global minimum values: Here is
    one example
    x*=(-10.0, 0.0)
    f(x*) = 0.0
    """
    global_minimum = bukin_n2_config['global_minimum']

    # Test the numpy version
    x = np.array([-10.0, 0.0])
    result = bukin_n2(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    result = bukin_n2(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    # Test the torch version
    x = torch.tensor([-10.0, 0.0], dtype=torch.float64)
    torch_result = bukin_n2(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-2)


def test_bukin_n4_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Ex Bukin N4 function has the correct global minimum.

    The function has many global minimum values: Here is
    one example
    x*=(-10.0, 0.0)
    f(x*) = 0.0
    """
    global_minimum = bukin_n4_config['global_minimum']

    # Test the numpy version
    x = np.array([-10.0, 0.0])
    result = bukin_n4(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    result = bukin_n4(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    # Test the torch version
    x = torch.tensor([-10.0, 0.0], dtype=torch.float64)
    torch_result = bukin_n4(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-2)


def test_camel_3hump_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Ex Camel 3 Hump function has the correct global minimum.

    The function has many global minimum values: Here is
    one example
    x*=(0, 0)
    f(x*) = 0.0
    """
    global_minimum = camel_3hump_config['global_minimum']

    # Test the numpy version
    x = np.array([0.0, 0.0])
    result = camel_3hump(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    result = camel_3hump(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    # Test the torch version
    x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    torch_result = camel_3hump(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-2)


def test_camel_6hump_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Ex Camel 6 Hump function has the correct global minimum.

    The function has many global minimum values: Here is
    one example
    x*=(-0.0898, 0.7126)
    x*=(0.0898, -0.7126)
    f(x*) = -1.0316
    """
    global_minimum = camel_6hump_config['global_minimum']

    # Test the numpy version
    x = np.array([-0.0898, 0.7126])
    result = camel_6hump(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    result = camel_6hump(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    # Test the torch version
    x = torch.tensor([0.0898, -0.7126], dtype=torch.float64)
    torch_result = camel_6hump(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-2)


def test_chung_reynolds_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Ex Chung Reynolds function has the correct global minimum.

    The function has many global minimum values: Here is
    one example
    x*=(0.0, 0.0)
    f(x*) = 0.0
    """
    global_minimum = chung_reynolds_config['global_minimum']

    # Test the numpy version
    x = np.array([0.0, 0.0])
    result = chung_reynolds(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    result = chung_reynolds(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    # Test the torch version
    x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    torch_result = chung_reynolds(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-2)


def test_cube_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Ex Cube function has the correct global minimum.

    The function has many global minimum values: Here is
    one example
    x*=(0.0, 0.0)
    f(x*) = 0.0

    x* given here:
    https://arxiv.org/pdf/1308.4008.pdf
    is incorrect
    """
    global_minimum = cube_config['global_minimum']

    # Test the numpy version
    x = np.array([1.0, 1.0])
    result = cube(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    result = cube(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    # Test the torch version
    x = torch.tensor([1.0, 1.0], dtype=torch.float64)
    torch_result = cube(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-2)


def test_xinsheyang_n2_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Ex Xin-She Yang N.2 function has the correct global minimum.

    The function has many global minimum values: Here is
    one example
    x*=(0.0, 0.0)
    f(x*) = 0.0

    x* given here:
    https://arxiv.org/pdf/1308.4008.pdf
    is incorrect
    """
    global_minimum = xinsheyang_n2_config['global_minimum']

    # Test the numpy version
    x = np.array([0.0, 0.0])
    result = xinsheyang_n2(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    result = xinsheyang_n2(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    # Test the torch version
    x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    torch_result = xinsheyang_n2(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-2)


def test_xinsheyang_n3_has_correct_global_minimum():
    """
    Function that tests if our implementation of the
    Ex Xin-She Yang N.3 function has the correct global minimum.

    The function has many global minimum values: Here is
    one example
    x*=(0.0, 0.0)
    f(x*) = 0.0

    x* given here:
    https://arxiv.org/pdf/1308.4008.pdf
    is incorrect
    """
    global_minimum = xinsheyang_n3_config['global_minimum']

    # Test the numpy version
    x = np.array([0.0, 0.0])
    result = xinsheyang_n3(x, version='numpy')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    result = xinsheyang_n3(x, version='pyomo')
    assert math.isclose(result, global_minimum, abs_tol=1e-2)

    # Test the torch version
    x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    torch_result = xinsheyang_n3(x, version='pytorch').numpy()
    assert math.isclose(torch_result, global_minimum, abs_tol=1e-2)
