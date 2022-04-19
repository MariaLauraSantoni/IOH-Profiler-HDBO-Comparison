from functools import partial
import random
from wrapper import marialaura
from my_logger import MyIOHFormatOnEveryEvaluationLogger, MyObjectiveFunctionWrapper
import sys
import os
import json
from ioh import Experiment, get_problem, logger, problem, OptimizationType
import numpy as np
import copy
import time
from datetime import timedelta


def decide_doe_size(dim):
    return 3 * dim


def decide_total_budget(dim, doe_size):
    return 5 * dim


class AlgorithmWrapper:
    def __init__(self, seed):
        self.opt = None
        self.seed = seed

    @staticmethod
    def __fitness_function_wrapper(x, f):
        if type(x) is np.ndarray:
            x = x.tolist()
        return f(x)

    @staticmethod
    def create_fitness(my_function):
        return partial(AlgorithmWrapper.__fitness_function_wrapper, f=my_function)

    def __call__(self, optimizer_name, f, fid, iid, dim):
        self.dim = dim
        self.optimizer_name = optimizer_name
        func = partial(AlgorithmWrapper.__fitness_function_wrapper, f=f)
        doe_size = decide_doe_size(self.dim)
        total_budget = decide_total_budget(self.dim, doe_size)
        self.opt = marialaura(
            optimizer_name, func, self.dim, total_budget, doe_size, self.seed)
        self.opt.run()


def run_particular_experiment(my_optimizer_name, fid, iid, dim, rep, folder_name):
    algorithm = AlgorithmWrapper(rep)
    l = MyIOHFormatOnEveryEvaluationLogger(
        folder_name=folder_name, algorithm_name=my_optimizer_name)
    print(f'    Logging to the folder {l.folder_name}')
    sys.stdout.flush()
    l.watch(algorithm, [])
    p = MyObjectiveFunctionWrapper(fid, iid, dim)
    p.attach_logger(l)
    algorithm(my_optimizer_name, p, fid, iid, dim)
    l.finish_logging()


def run_experiment():
    if len(sys.argv) == 1:
        print('No configs given')
        return
    with open(sys.argv[1]) as f:
        m = json.load(f)
    print(f'Running with config {m} ...')
    start = time.time()
    run_particular_experiment(
        m['opt'], m['fid'], m['iid'], m['dim'], m['seed'], m['folder'])
    end = time.time()
    sec = int(round(end - start))
    x = str(timedelta(seconds=sec)).split(':')
    print(
        f'    Done in {sec} seconds. Which is {x[0]} hours, {x[1]} minutes and {x[2]} seconds')


if __name__ == '__main__':
    run_experiment()
