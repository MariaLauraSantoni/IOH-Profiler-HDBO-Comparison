import numpy as np

from ioh import get_problem, logger


class SaasboWrapper:
    def __init__(self, func, dim, ub, lb, total_budget, DoE_size, random_seed):
        import sys
        sys.path.append('./mylib/' + 'lib_' + "saasbo")
        print(sys.path)

        self.func = func
        self.dim = dim
        self.ub = ub
        self.lb = lb
        self.total_budget = total_budget
        self.Doe_size = DoE_size
        self.random_seed = random_seed

    def run(self):
        from saasbo import run_saasbo
        run_saasbo(
            self.func,
            np.ones(self.dim) * self.ub,
            np.ones(self.dim) * self.lb,
            self.total_budget,
            self.Doe_size,
            self.random_seed,
            alpha=0.01,
            num_warmup=256,
            num_samples=256,
            thinning=32,
            device="cpu",
        )

class BO_sklearnWrapper:
    def __init__(self, func, dim, ub, lb, total_budget, DoE_size, random_seed):
        import sys
        sys.path.append('./mylib/' + 'lib_' + "BO_sklearn")
        print(sys.path)

        self.func = func
        self.dim = dim
        self.ub = ub
        self.lb = lb
        self.total_budget = total_budget
        self.Doe_size = DoE_size
        self.random_seed = random_seed

    def run(self):
        from skopt import gp_minimize

        gp_minimize(self.func,  # the function to minimize
                    list((((self.lb, self.ub),) * self.dim)),  # the bounds on each dimension of x
                    acq_func="EI",  # the acquisition function
                    n_calls=self.total_budget,  # the number of evaluations of f
                    n_random_starts=self.Doe_size,  # the number of random initialization points
                    noise=0.1 ** 2,  # the noise level (optional)
                    random_state=1234)

class BO_bayesoptimWrapper:
    def __init__(self, func, dim, ub, lb, total_budget, DoE_size, random_seed):
        import sys
        sys.path.append('./mylib/' + 'lib_' + "BO_bayesoptim")
        print(sys.path)

        self.func = func
        self.dim = dim
        self.ub = ub
        self.lb = lb
        self.total_budget = total_budget
        self.Doe_size = DoE_size
        self.random_seed = random_seed

    def run(self):
        from bayes_optim import BO, RealSpace
        from bayes_optim.surrogate import GaussianProcess

        space = RealSpace([self.lb, self.ub]) * self.dim  # create the search space

        # hyperparameters of the GPR model
        thetaL = 1e-10 * (self.ub - self.lb) * np.ones(self.dim)
        thetaU = 10 * (self.ub - self.lb) * np.ones(self.dim)
        model = GaussianProcess(  # create the GPR model
            thetaL=thetaL, thetaU=thetaU
        )

        opt = BO(
            search_space=space,
            obj_fun=self.func,
            model=model,
            DoE_size=self.Doe_size,  # number of initial sample points
            max_FEs=self.total_budget,  # maximal function evaluation
            verbose=True
        )
        opt.run()

class randomWrapper:
    def __init__(self, func, dim, ub, lb, total_budget, DoE_size, random_seed):

        self.func = func
        self.dim = dim
        self.ub = ub
        self.lb = lb
        self.total_budget = total_budget
        self.Doe_size = DoE_size
        self.random_seed = random_seed

    def run(self):
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

        random_search(self.func, self.random_seed, self.total_budget)

class linearPCABOWrapper:
    def __init__(self, func, dim, ub, lb, total_budget, DoE_size, random_seed):
        import sys
        sys.path.append('./mylib/' + 'lib_' + "linearPCABO")
        print(sys.path)

        self.func = func
        self.dim = dim
        self.ub = ub
        self.lb = lb
        self.total_budget = total_budget
        self.Doe_size = DoE_size
        self.random_seed = random_seed

    def run(self):
        import sys
        sys.path.insert(0, "./mylib/lib_linearPCABO/Bayesian-Optimization")
        from bayes_optim.extension import PCABO, RealSpace

        np.random.seed(123)
        space = RealSpace([self.lb, self.ub]) * self.dim
        opt = PCABO(
            search_space=space,
            obj_fun=self.func,
            DoE_size=self.Doe_size,
            max_FEs=self.total_budget,  # 10 * dim + 50,
            verbose=True,
            n_point=1,
            n_components=0.95,
            acquisition_optimization={"optimizer": "BFGS"},
        )
        print(opt.run())

class turbo1Wrapper:
    def __init__(self, func, dim, ub, lb, total_budget, DoE_size, random_seed):
        import sys
        sys.path.append('./mylib/' + 'lib_' + "turbo1")
        print(sys.path)

        self.func = func
        self.dim = dim
        self.ub = ub
        self.lb = lb
        self.total_budget = total_budget
        self.Doe_size = DoE_size
        self.random_seed = random_seed

    def run(self):
        from turbo import Turbo1
        import torch
        import math
        import matplotlib
        import matplotlib.pyplot as plt
        turbo1 = Turbo1(
            f=self.func,  # Handle to objective function
            lb=np.ones(dim) * self.lb,  # Numpy array specifying lower bounds
            ub=np.ones(dim) * self.ub,  # Numpy array specifying upper bounds
            n_init=self.Doe_size,  # Number of initial bounds from an Latin hypercube design
            max_evals=self.total_budget,  # Maximum number of evaluations
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
class turbomWrapper:
    def __init__(self, func, dim, ub, lb, total_budget, DoE_size, random_seed):
        import sys
        sys.path.append('./mylib/' + 'lib_' + "turbom")
        print(sys.path)

        self.func = func
        self.dim = dim
        self.ub = ub
        self.lb = lb
        self.total_budget = total_budget
        self.Doe_size = DoE_size
        self.random_seed = random_seed

    def run(self):
        from turbo import TurboM
        import torch
        import math
        import matplotlib
        import matplotlib.pyplot as plt
        tr = math.floor(self.total_budget / self.Doe_size) - 1

        turbo_m = TurboM(
            f=self.func,  # Handle to objective function
            lb=np.ones(dim) * self.lb,  # Numpy array specifying lower bounds
            ub=np.ones(dim) * self.ub,  # Numpy array specifying upper bounds
            n_init=self.Doe_size,  # Number of initial bounds from an Symmetric Latin hypercube design
            max_evals=self.total_budget,  # Maximum number of evaluations
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


def marialaura(optimizer_name, func, ml_dim, ml_total_budget, ml_DoE_size, random_seed):
    ub = +5
    lb = -5
    if optimizer_name == "saasbo":
        solver = SaasboWrapper(func=func, dim=ml_dim, ub=ub, lb=lb, total_budget=ml_total_budget, DoE_size=ml_DoE_size,
                               random_seed=random_seed)
        solver.run()
    if optimizer_name == "BO_sklearn":
        solver = BO_sklearnWrapper(func=func, dim=ml_dim, ub=ub, lb=lb, total_budget=ml_total_budget, DoE_size=ml_DoE_size,
                               random_seed=random_seed)
        solver.run()
    if optimizer_name == "BO_bayesoptim":
        solver = BO_bayesoptimWrapper(func=func, dim=ml_dim, ub=ub, lb=lb, total_budget=ml_total_budget, DoE_size=ml_DoE_size,
                               random_seed=random_seed)
        solver.run()
    if optimizer_name == "random":
        solver = randomWrapper(func=func, dim=ml_dim, ub=ub, lb=lb, total_budget=ml_total_budget, DoE_size=ml_DoE_size,
                               random_seed=random_seed)
        solver.run()
    if optimizer_name == "linearPCABO":
        solver = randomWrapper(func=func, dim=ml_dim, ub=ub, lb=lb, total_budget=ml_total_budget, DoE_size=ml_DoE_size,
                               random_seed=random_seed)
        solver.run()
    if optimizer_name == "turbo1":
        solver = turbomWrapper(func=func, dim=ml_dim, ub=ub, lb=lb, total_budget=ml_total_budget, DoE_size=ml_DoE_size,
                               random_seed=random_seed)
    if optimizer_name == "turbom":
        solver = turbomWrapper(func=func, dim=ml_dim, ub=ub, lb=lb, total_budget=ml_total_budget, DoE_size=ml_DoE_size,
                               random_seed=random_seed)


if __name__ == "__main__":
    dim = 10
    total_budget = 20
    doe_size = 5
    seed = 0
    # Algorithm alternatives:
    algorithm_name = "turbo1"

    f = get_problem(21, dimension=dim, instance=1, problem_type='Real')

    marialaura(algorithm_name, f, dim, total_budget, doe_size, seed)
