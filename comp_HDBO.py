# -*- coding: utf-8 -*-

#Import the ioh logger module
from ioh import get_problem, logger
import numpy as np

# Set solver
# Alternatives: "saasbo", "BO_sklearn", "BO_bayesoptim", "random", "RPA_BO", "linearPCABO"
solver = "linearPCABO"

import sys
sys.path.append('./mylib/' + 'lib_' + solver)

print(sys.path)





# Common settings
dim = 10
ub = 5
lb = -5
budget=10
#budget= 10 * dim + 50
DoE_samples = int(.20 * budget)


#Create a problem object, either by giving the problem id from within the suite
f = get_problem(21, dimension=dim, instance=1, problem_type = 'Real')

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
        # packages for saasbo
        from saasbo import run_saasbo

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
        # packages for BO_sklearn
        from skopt import gp_minimize

        gp_minimize(f,  # the function to minimize
                    list((((lb, ub),) * dim)),  # the bounds on each dimension of x
                    acq_func="EI",  # the acquisition function
                    n_calls=budget,  # the number of evaluations of f
                    n_random_starts=DoE_samples,  # the number of random initialization points
                    noise=0.1 ** 2,  # the noise level (optional)
                    random_state=1234)
    elif solver == "linearPCABO":
        # packages for linearPCABO
        sys.path.insert(0, "./mylib/lib_linearPCABO/Bayesian-Optimization")
        from bayes_optim.extension import PCABO, RealSpace

        np.random.seed(123)
        space = RealSpace([lb, ub]) * dim
        opt = PCABO(
            search_space=space,
            obj_fun=f,
            DoE_size=3, # DoE_samples,
            max_FEs=10, # 10 * dim + 50,
            verbose=True,
            n_point=1,
            n_components=0.95,
            acquisition_optimization={"optimizer": "BFGS"},
        )
        print(opt.run())


