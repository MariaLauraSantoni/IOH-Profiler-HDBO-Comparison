# -*- coding: utf-8 -*-

#Import the ioh logger module
from ioh import get_problem, logger
import numpy as np

# Set solver
# Alternatives: "saasbo", "BO_sklearn", "BO_bayesoptim", "random", "RPA_BO", "linearPCABO"
solver = "saasbo"

import sys
sys.path.append('./mylib/' + 'lib_' + solver)

print(sys.path)

# packages for saasbo
from saasbo import run_saasbo
# packages for BO_sklearn
from skopt import gp_minimize


# Common settings
dim = 10
ub = 5
lb = -5
budget=10
#budget= 10 * dim + 50
DoE_samples = int(.20 * budget)


#Create a problem object, either by giving the problem id from within the suite
f = get_problem(7, dimension=dim, instance=1, problem_type = 'Real')

#Print some properties of the problem
print(f.meta_data)
#Access the box-constrains for this problem
print(f.constraint.lb, f.constraint.ub)
#Show the state of the optimization
print(f.state)

#Create default logger compatible with IOHanalyzer
l = logger.Analyzer(root="data", folder_name="run", algorithm_name=solver, algorithm_info="test of IOHexperimenter in python")
f.attach_logger(l)


if __name__ == "__main__":
    # # create an instance of this algorithm
    # o = RandomSearch(1000)
    f.attach_logger(l)
    # saasbo(f)
    if solver == "saasbo":
        run_saasbo(
            f,
            np.ones(dim)*ub,
            np.ones(dim)*lb,
            budget,
            DoE_samples,
            seed=42,
            alpha=0.01,
            num_warmup=256,
            num_samples=256,
            thinning=32,
            device="cpu",
        )
    elif solver == "BO_sklearn":
        gp_minimize(f,  # the function to minimize
                    list((((lb, ub),) * dim)),  # the bounds on each dimension of x
                    acq_func="EI",  # the acquisition function
                    n_calls=budget,  # the number of evaluations of f
                    n_random_starts=DoE_samples,  # the number of random initialization points
                    noise=0.1 ** 2,  # the noise level (optional)
                    random_state=1234)
