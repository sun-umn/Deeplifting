# Polynomial problem instances for testing SOS


def booth(x1, x2):
    p = (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2
    bounds = [x1 - 10, x1 + 10, x2 - 10, x2 + 10]
    deg = 2
    return {'p': p, 'bounds': bounds, 'deg': deg}


def matyas(x1, x2):
    p = 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2
    bounds = [x1 - 10, x1 + 10, x2 - 10, x2 + 10]
    deg = 2
    return {'p': p, 'bounds': bounds, 'deg': deg}


def three_hump_camel(x1, x2):
    p = 2 * x1**2 - 1.05 * x1**4 + (1 / 6) * x1**6 + x1 * x2 + x2**2
    bounds = [x1 - 5, x1 + 5, x2 - 5, x2 + 5]
    deg = 6
    return {'p': p, 'bounds': bounds, 'deg': deg}


def motzkin(x1, x2):
    p = x1**4 * x2**2 + x1**2 * x2**4 - 3 * x1**2 * x2**2 + 1
    bounds = [x1 - 2, x1 + 2, x2 - 2, x2 + 2]
    deg = 6
    return {'p': p, 'bounds': bounds, 'deg': deg}


def styblinzki_tang(x1, x2):
    p = (0.5 * x1**4 - 8 * x1**2 + 2.5 * x1) + (
        0.5 * x2**4 - 8 * x2**2 + 2.5 * x2
    )
    bounds = [x1 - 5, x1 + 5, x2 - 5, x2 + 5]
    deg = 4
    return {'p': p, 'bounds': bounds, 'deg': deg}


# Having issues !!
def rosenbrock(x1, x2):
    p = 100 * (x2 - x1**2) ** 2 + (x1 - 1) ** 2
    bounds = [x1 - 2.048, x1 + 2.048, x2 - 2.048, x2 + 2.048]
    deg = 4
    return {'p': p, 'bounds': bounds, 'deg': deg}


# Having issues !!
def matyas_modified_s(x1, x2):
    p = 0.26 * ((20 * x1 - 10) ** 2 + (20 * x2 - 10) ** 2) - 0.48 * (20 * x1 - 10) * (
        20 * x2 - 10
    )
    bounds = [x1, x2, 1 - (x1 + x2)]
    deg = 2
    return {'p': p, 'bounds': bounds, 'deg': deg}
