import numpy as np
np.random.seed(237)
import matplotlib.pyplot as plt
from skopt.plots import plot_gaussian_process

from gp_experiment_runner import load_dataset

from ioh import get_problem, logger

# Common settings
dim = 10
ub = 5
lb = -5

budget=10
#budget= 10 * dim + 50
DoE_samples = int(.20 * budget)

#Create a problem object, either by giving the problem id from within the suite
f = get_problem(7, dimension=dim, instance=1, problem_type = 'Real')


from skopt import gp_minimize

res = gp_minimize(f,                  # the function to minimize
                  [(lb , ub)],      # the bounds on each dimension of x
                  acq_func="EI",      # the acquisition function
                  n_calls=15,         # the number of evaluations of f
                  n_random_starts=5,  # the number of random initialization points
                  noise=0.1**2,       # the noise level (optional)
                  random_state=1234,  # the random seed
                  base_estimator="RPA")
