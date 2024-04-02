# Comparison of High-Dimensional Bayesian Optimization Algorithms on BBOB
This repository contains the code used to generate the results in the paper Comparison of High-Dimensional Bayesian Optimization Algorithms on BBOB.

It proposes a modular algorithm framework to make the implementation of several algorithms compared within the paper compatible
via [IOHprofiler](https://iohprofiler.github.io/), and appropriate code to store all the data obtained.
The selected algorithms are: 
- Vanilla Bayesian Optimization, taken from the Python module [scikit-optimize](https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html).
- CMA-ES from the [pycma](https://github.com/CMA-ES/pycma) package.
- Random search, taken from the Python module numpy using the method [random.uniform](https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html).
- SAASBO algorithm from [High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces](https://arxiv.org/pdf/2103.00349.pdf).
- RDUCB introduced in [Are Random Decompositions all we need in High Dimensional Bayesian Optimisation?](https://arxiv.org/abs/2301.12844).
- PCA-BO proposed in [High Dimensional Bayesian Optimization Assisted by Principal Component Analysis](https://arxiv.org/pdf/2007.00925.pdf).
- KPCA-BO introduced in [High Dimensional Bayesian Optimization with Kernel Principal Component Analysis](https://arxiv.org/pdf/2204.13753.pdf).
- TuRBO from [Scalable Global Optimization via Local Bayesian Optimization](https://proceedings.neurips.cc/paper/2019/file/6c990b7aca7bc7058f5e98ea909e924b-Paper.pdf).

This code compares these approaches on the 24 functions of the Black-Box Optimization Benchmarking (BBOB) suite from the [COCO](https://arxiv.org/pdf/1603.08785.pdf) benchmarking environment suite using their definition from [IOHprofiler](https://iohprofiler.github.io/). It is based on the original repositories and modules of the selected algorithms [vanilla Bayesian Optimization](https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html), [CMA-ES](https://github.com/CMA-ES/pycma), [random search](https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html), [SAASBO](https://github.com/martinjankowiak/saasbo), [RDUCB](https://github.com/huawei-noah/HEBO/tree/master/RDUCB), [PCA-BO](https://github.com/wangronin/Bayesian-Optimization/tree/KPCA-BO), [KPCA-BO](https://github.com/wangronin/Bayesian-Optimization/tree/KPCA-BO) and [TuRBO](https://github.com/uber-research/TuRBO). We provide all the Python files to run the paper experiments and to store results in data files.

# Libraries and dependencies

The implementation of all tasks and algorithms to perform experiments are in Python 3.7.4 and all the libraries used are listed in `requirements.txt`.

# Structure
- `run_experiment.py` is the main file, used to run any experiments. It initializes the main setting of the experiment, calls the chosen algorithm, and writes log files. It takes as argument a file `.json` that is the output of the file `gen_config.py`.
- `wrapper.py` contains the definition of all algorithms and the method `wrapopt` that runs the main loop of the chosen algorithm. It is called by the file `run_experiment.py`.
- `my_logger.py` defines all the functions needed to generate the files to store data output by a run. It is called by the file `run_experiment.py`.
- `total_config.json` allows defining the settings of the experiment and it has to be the argument of the file `gen_config.py`. 
- `gen_config.py` generates a folder called `config` containing files to run each algorithm with the parameters chosen in `total_config.json` given as an input and a bash script to run experiments with a Slurm job scheduler.
- `mylib` contains one folder for each algorithm with all the classes and functions needed to run them.
- `Bayesian-Optimization.zip` contains the cloned repository [Bayesian-Optimization](https://github.com/wangronin/Bayesian-Optimization/tree/KPCA-BO) with little changes to track the CPU time for the algorithms PCA-BO and KPCA-BO.
- `RDUCB.zip` contains the cloned repository [RDUCB](https://github.com/huawei-noah/HEBO/tree/master/RDUCB) with little changes to track the CPU time for the algorithm RDUCB.
- `Gpy.zip` and `GpyOpt.zip` contain the modules Gpy and GpyOpt, respectively, with little changes to track the CPU time for the algorithm RDUCB.
- `skopt.zip` contains the module skopt with little changes to track the CPU time for the algorithm vanilla Bayesian Optimization.
- `requirements.txt` contains the list of all the project’s dependencies with the specific version of each dependency.

# Execution from source
## Dependencies to run from source

Running this code from source requires Python 3.7.4, and the libraries given in `requirements.txt` (Warning: preferably use a virtual environment for this specific project, to avoid breaking the dependencies of your other projects). In Ubuntu, installing the dependencies can be done using the following command:

```
pip install -r requirements.txt
```

## Specific modules to copy for tracking the CPU time in the log file
To correctly track the CPU time this project needs some modified modules and a modified cloned repository. Follow the steps below:

1. Unzip the folders `skopt.zip`, `Bayesian-Optimization.zip`, `Gpy.zip`, `GpyOpt.zip` and `RDUCB.zip`:
```
unzip skopt.zip
unzip Bayesian-Optimization.zip
unzip Gpy.zip
unzip GpyOpt.zip
unzip RDUCB.zip
```
2. Find the path of the used Python site-packages directory:
```
python -m site
```
3. Move `skopt`, `GPy` and `GPyOPt` to the used Python site-packages directory:
```
mv skopt "found_path_site_packages"
mv Gpy "found_path_site_packages"
mv GpyOPt "found_path_site_packages"
```
4. Move `Bayesian-Optimization` and `RDUCB` to the right libraries inside the project:
```
mv Bayesian-Optimization mylib/lib_BO_bayesoptim
mv RDUCB mylib/lib_RDUCB/HEBO
```
## Run from source
First of all, the parameters of the experiment need to be decided in the file `total_config.json`: 
- `folder` is the first part of the name of the folders that will contain all the result data from the experiment. The number of the folders for each function indicated in `fiids` that will be generated to store the results are indicated in `reps`.
- `optimizers` is the name of the algorithm used during the experiment. It can be chosen among `BO_sklearn`, `pyCMA`, `random`, `saasbo`, `RDUCB`, `linearPCABO`, `KPCABO`, `turbo1` and `turbom`.
- `fiids` defines which functions the algorithm has to optimize. It can be a single number or multiple numbers separated by a comma in the range of the 24 BBOB functions.
- `iids` is the number of the problem instance, in the paper 0, 1, and 2 are performed.
- `dims` is the dimension of the problem.
- `reps` is the number of run repetitions with the same settings (optimizer, function id, instance id, etc.). Different repetitions differ only for the seed. Inside the folder containing the results, a `config` folder will be generated containing `reps` .json files, one for each repetition. The number at the beginning of the `.json` file represents the seed number for that specific repetition, starting from 0 (ex. 0.json stores the settings for running an experiment using seed 0). 
- `lb` and `ub` are the lower bound and the upper bound of the design domain. In the paper, they are fixed as -5 and 5, respectively.
- `extra` contains extra text information to store in the result folder.
### Execute repetitions in parallel using a cluster
If a job scheduling system for Linux clusters is available, the batch script can be edited inside the file `gen_config.py`. 
After choosing the parameters and editing the batch script, a folder called `run_current_date_and_time` containing folders with the result data and the `config` folder will be generated using the following command:
```
python gen_config.py total_config.json
```
and the jobs can be launched by typing the last command line that will appear as screen output.
### Execute a single run
Here, there is no need to adjust the settings to generate the batch script editing the file `gen_config.py`. Therefore, after choosing the parameters the folder called `run_current_date_and_time` containing the folders with the result data, and the `config` folder will be generated using the following command:
```
python gen_config.py total_config.json
```
then, move to the folder `run_current_date_and_time` typing the first half of the last command line that will appear as screen output (the part before &&).
A single run with a specific number of seeds (till reps-1) can be executed using the following command:
```
python ../run_experiment.py config/number_of_seeds.json
```
## Analysis from source
Reps-folders for each function indicated in `fiids` with the first part of the name stored in `folder` inside the file `total_config.json` will be generated inside the folder `run_current_date_and_time`. Each of them contains a folder `data_number_and_name_of_the_function` that stores a `.dat` file with all the results about the loss and the different CPU times tracked (the loss examined in the paper is stored under the name `best-so-far f(x)`).


