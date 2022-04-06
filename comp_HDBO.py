# -*- coding: utf-8 -*-

#Import the ioh logger module
from ioh import get_problem, logger
import numpy as np


# Set solver
# Alternatives: "saasbo", "BO_sklearn", "BO_bayesoptim", "random", "linearPCABO", "turbo1", "turbom"


#alternatives = [ "saasbo", "BO_sklearn", "BO_bayesoptim", "random", "linearPCABO" ]
alternatives = ["SMAC3"]



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


for item in alternatives :

    solver = item

    import sys

    sys.path.append('./mylib/' + 'lib_' + solver)

    print(sys.path)
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

        elif solver == "BO_bayesoptim":
            # packages for BO_bayesoptim
            from bayes_optim import BO, RealSpace
            from bayes_optim.surrogate import GaussianProcess

            space = RealSpace([lb, ub]) * dim  # create the search space

            # hyperparameters of the GPR model
            thetaL = 1e-10 * (ub - lb) * np.ones(dim)
            thetaU = 10 * (ub - lb) * np.ones(dim)
            model = GaussianProcess(  # create the GPR model
                thetaL=thetaL, thetaU=thetaU
            )

            opt = BO(
                search_space=space,
                obj_fun=f,
                model=model,
                DoE_size=DoE_samples,  # number of initial sample points
                max_FEs=budget,  # maximal function evaluation
                verbose=True
            )
            opt.run()

        elif solver == "random":
            # packages for random
            import ioh
            def random_search(problem: ioh.problem.Real, seed: int = 42, budget: int = None) -> ioh.RealSolution:
                np.random.seed(seed)

                if budget is None:
                    budget = int(problem.meta_data.n_variables * 1e4)

                for _ in range(budget):
                    x = np.random.uniform(problem.constraint.lb, problem.constraint.ub)

                    # problem automatically tracks the current best search point
                    f = problem(x)

                return problem.state.current_best


            random_search(f, 42, budget)

        elif solver == "linearPCABO":
            # packages for linearPCABO
            sys.path.insert(0, "./mylib/lib_linearPCABO/Bayesian-Optimization")
            from bayes_optim.extension import PCABO, RealSpace

            np.random.seed(123)
            space = RealSpace([lb, ub]) * dim
            opt = PCABO(
                search_space=space,
                obj_fun=f,
                DoE_size= DoE_samples,
                max_FEs=budget, # 10 * dim + 50,
                verbose=True,
                n_point=1,
                n_components=0.95,
                acquisition_optimization={"optimizer": "BFGS"},
            )
            print(opt.run())

        elif solver == "turbo1":
            from turbo import Turbo1
            import torch
            import math
            import matplotlib
            import matplotlib.pyplot as plt

            turbo1 = Turbo1(
                f=f,  # Handle to objective function
                lb=np.ones(dim)*lb,  # Numpy array specifying lower bounds
                ub=np.ones(dim)*ub,  # Numpy array specifying upper bounds
                n_init=DoE_samples,  # Number of initial bounds from an Latin hypercube design
                max_evals=budget,  # Maximum number of evaluations
                batch_size=5,  # How large batch size TuRBO uses
                verbose=True,  # Print information from each batch
                use_ard=True,  # Set to true if you want to use ARD for the GP kernel
                max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
                n_training_steps=50,  # Number of steps of ADAM to learn the hypers
                min_cuda=1024,  # Run on the CPU for small datasets
                device="cpu",  # "cpu" or "cuda"
                dtype="float64",  # float64 or float32
            )
            turbo1.optimize()

        elif solver == "turbom":
            from turbo import TurboM
            import torch
            import math
            import matplotlib
            import matplotlib.pyplot as plt
            tr = math.floor(budget/DoE_samples) -1

            turbo_m = TurboM(
                f=f,  # Handle to objective function
                lb=np.ones(dim)*lb,  # Numpy array specifying lower bounds
                ub=np.ones(dim)*ub,  # Numpy array specifying upper bounds
                n_init=DoE_samples,  # Number of initial bounds from an Symmetric Latin hypercube design
                max_evals=budget,  # Maximum number of evaluations
                n_trust_regions=tr,  # Number of trust regions
                batch_size=5,  # How large batch size TuRBO uses
                verbose=True,  # Print information from each batch
                use_ard=True,  # Set to true if you want to use ARD for the GP kernel
                max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
                n_training_steps=50,  # Number of steps of ADAM to learn the hypers
                min_cuda=1024,  # Run on the CPU for small datasets
                device="cpu",  # "cpu" or "cuda"
                dtype="float64",  # float64 or float32
            )
            turbo_m.optimize()
        elif solver == "SMAC3":
            from ConfigSpace.hyperparameters import UniformFloatHyperparameter

            # Import ConfigSpace and different types of parameters
            from smac.configspace import ConfigurationSpace
            from smac.facade.smac_bb_facade import SMAC4BB
            from smac.optimizer.acquisition import EI

            # Import SMAC-utilities
            from smac.scenario.scenario import Scenario

            # Build Configuration Space which defines all parameters and their ranges
            cs = ConfigurationSpace()
            x0 = UniformFloatHyperparameter("x0", -5, 10, default_value=-3)
            x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=-4)
            cs.add_hyperparameters([x0, x1])

            # Scenario object
            scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                                 "runcount-limit": budget,  # max. number of function evaluations
                                 "cs": cs,  # configuration space
                                 "deterministic": True
                                 })

            # Use 'gp' or 'gp_mcmc' here
            model_type = 'gp'

            import smac  # noqa  # pylint: disable=unused-importa
            import scipy.optimize  # noqa  # pylint: disable=unused-importa
            from smac.facade.func_facade import fmin_smac  # noqa  # pylint: disable=unused-import

            x, cost, _ = fmin_smac(
                func=f, x0=[0.0] * dim, bounds=[(lb, ub)] * dim, maxfun=budget
            )  # Passing a seed makes fmin_smac determistic

            smac = SMAC4BB(scenario=scenario,
                           model_type=model_type,
                           rng=np.random.RandomState(42),
                           acquisition_function=EI,  # or others like PI, LCB as acquisition functions
                           tae_runner=f)

            smac.optimize()



