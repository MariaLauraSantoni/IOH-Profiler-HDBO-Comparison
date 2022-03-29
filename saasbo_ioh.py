# -*- coding: utf-8 -*-

import sys

sys.path.append('.')
sys.path.append('lib')

# import argparse
            
import numpy as np
#import numpyro
#from numpyro.util import enable_x64
from saasbo import run_saasbo
import ioh
#Import the ioh logger module
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

#Print some properties of the problem
print(f.meta_data)
#Access the box-constrains for this problem
print(f.constraint.lb, f.constraint.ub)
#Show the state of the optimization
print(f.state)

#Create default logger compatible with IOHanalyzer
l = logger.Analyzer(root="data", folder_name="run", algorithm_name="saasbo", algorithm_info="test of IOHexperimenter in python")
f.attach_logger(l)


#[a, b]= run_saasbo(
#        f,
#        np.ones(dim)*ub,
#        np.ones(dim)*lb,
#        budget,
#        DoE_samples,
#        seed=42,
#        alpha=0.01,
#        num_warmup=256,
#        num_samples=256,
#        thinning=32,
#        device="cpu",
#    )
#
#def saasbo(problem: ioh.problem.Real, seed: int = 42, budget: int = budget) -> ioh.RealSolution:
#    np.random.seed(seed)
#    return min(b)
        


if __name__ == "__main__":
    # # create an instance of this algorithm
    # o = RandomSearch(1000)
    f.attach_logger(l)
    # saasbo(f)
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
