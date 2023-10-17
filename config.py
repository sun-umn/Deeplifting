# In this file we will list the different problems
# and the configurations for deeplifting

# 2D test problems
# Identify problems to run
low_dimensional_problem_names = [
    'ackley',
    'ackley2',
    'ackley3',
    'adjiman',
    'alpine1',
    'alpine2',
    'bartels_conn',
    'beale',
    'bird',
    'bohachevsky1',
    'bohachevsky2',
    'bohachevsky3',
    'booth',
    'branin_rcos',
    'brent',
    'bukin_n2',
    'bukin_n4',
    'bukin_n6',
    'camel_3hump',
    'camel_6hump',
    'chung_reynolds',
    'cross_in_tray',
    'cross_leg_table',
    'crowned_cross',
    'drop_wave',
    'eggholder',
    'griewank',
    'holder_table',
    'levy',
    'levy_n13',
    'mathopt6',
    'rastrigin',
    'rosenbrock',
    'schaffer_n2',
    'schaffer_n4',
    'schwefel',
    'shubert',
    'rosenbrock',
    'xinsheyang_n2',
    'xinsheyang_n3',
]

# High dimension test problems
# Ackley Series
ackley_series = [
    'ackley_3d',
    'ackley_5d',
    'ackley_30d',
    'ackley_100d',
    'ackley_500d',
    'ackley_1000d',
]

# Alpine1 Series - Origin Solution
alpine_series = [
    'alpine1_3d',
    'alpine1_5d',
    'alpine1_30d',
    'alpine1_100d',
    'alpine1_500d',
    'alpine1_1000d',
]

# Chung-Reynolds Series - Origin Solution
chung_reynolds_series = [
    'chung_reyonlds_3d',
    'chung_reynolds_5d',
    'chung_reynolds_30d',
    'chung_reynolds_100d',
    'chung_reynolds_500d',
    'chung_reynolds_1000d',
]

# Griewank Series - Origin Solution
griewank_series = [
    'griewank_3d',
    'griewank_5d',
    'griewank_30d',
    'griewank_100d',
    'griewank_500d',
    'griewank_1000d',
]

# Lennard jones series
lennard_jones_series = [
    'lennard_jones_6d',
    'lennard_jones_9d',
    'lennard_jones_12d',
    'lennard_jones_15d',
    'lennard_jones_18d',
    'lennard_jones_21d',
    'lennard_jones_24d',
    'lennard_jones_27d',
    'lennard_jones_30d',
    'lennard_jones_39d',
    'lennard_jones_42d',
    'lennard_jones_45d',
]

# Levy Series - Non-origin solution
levy_series = [
    'levy_3d',
    'levy_5d',
    'levy_30d',
    'levy_100d',
    'levy_500d',
    'levy_1000d',
]

# Qing Series - Non-origin solution
qing_series = [
    'qing_3d',
    'qing_5d',
    'qing_30d',
    'qing_100d',
    'qing_500d',
    'qing_1000d',
]

# Rastrigin series - Origin solution
rastrigin_series = [
    'rastrigin_3d',
    'rastrigin_5d',
    'rastrigin_30d',
    'rastrigin_100d',
    'rastrigin_500d',
    'rastrigin_1000d',
]

# Schwefel series - non-origin solution
schwefel_series = [
    'schwefel_3d',
    'schwefel_5d',
    'schwefel_30d',
    'schwefel_100d',
    'schwefel_500d',
    'schwefel_1000d',
]

# High dimensional problems list
high_dimensional_problem_names = (
    ackley_series
    + alpine_series
    + chung_reynolds_series
    + griewank_series
    + lennard_jones_series
    + levy_series
    + qing_series
    + rastrigin_series
    + schwefel_series
)

# Configurations for deeplifting
# Identify available hidden sizes
hidden_size_64 = (64,)
hidden_size_128 = (128,)
hidden_size_256 = (256,)
hidden_size_384 = (384,)
hidden_size_512 = (512,)
hidden_size_768 = (768,)
hidden_size_1024 = (1024,)
hidden_size_2048 = (2048,)

# Hidden size combinations
search_hidden_sizes = [
    # Hidden sizes of 128
    hidden_size_128 * 2,
    hidden_size_128 * 3,
    hidden_size_128 * 4,
    hidden_size_128 * 5,
    hidden_size_128 * 10,
    hidden_size_128 * 20,
    # Hidden sizes of 256
    hidden_size_256 * 2,
    hidden_size_256 * 3,
    hidden_size_256 * 4,
    hidden_size_256 * 5,
    hidden_size_256 * 10,
    hidden_size_256 * 20,
    # Hidden sizes of 382
    hidden_size_384 * 2,
    hidden_size_384 * 3,
    hidden_size_384 * 4,
    hidden_size_384 * 5,
    hidden_size_384 * 10,
    hidden_size_384 * 20,
    # Hidden sizes of 512
    hidden_size_512 * 2,
    hidden_size_512 * 3,
    hidden_size_512 * 4,
    hidden_size_512 * 5,
    hidden_size_512 * 10,
    hidden_size_512 * 20,
]

# Input sizes
search_input_sizes = [1]

# Hidden activations
search_hidden_activations = ['sine']

# Ouput activations
search_output_activations = ['sine']

# Aggregate functions - for skip connections
search_agg_functions = ['sum']

# Include BN
search_include_bn = [False, True]
