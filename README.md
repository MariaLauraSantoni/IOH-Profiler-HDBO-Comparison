# Comparison of High-Dimensional Bayesian Optimization Algorithms on BBOB
This repository contains the code used to generate the results in the paper Comparison of High-Dimensional Bayesian Optimization Algorithms on BBOB.

It proposes a modular framework to make the implementation of several algorithms compared within the paper compatible
with [IOHprofiler](https://iohprofiler.github.io/) and log their performance.

The compared algorithms are: 
- Vanilla Bayesian Optimization, taken from the Python module [scikit-optimize](https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html).
- CMA-ES from the [pycma](https://github.com/CMA-ES/pycma) package.
- Random search, taken from the Python module numpy using the method [random.uniform](https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html).
- SAASBO algorithm from [High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces](https://arxiv.org/pdf/2103.00349.pdf).
- RDUCB introduced in [Are Random Decompositions all we need in High Dimensional Bayesian Optimisation?](https://arxiv.org/abs/2301.12844).
- PCA-BO proposed in [High Dimensional Bayesian Optimization Assisted by Principal Component Analysis](https://arxiv.org/pdf/2007.00925.pdf).
- KPCA-BO introduced in [High Dimensional Bayesian Optimization with Kernel Principal Component Analysis](https://arxiv.org/pdf/2204.13753.pdf).
- TuRBO from [Scalable Global Optimization via Local Bayesian Optimization](https://proceedings.neurips.cc/paper/2019/file/6c990b7aca7bc7058f5e98ea909e924b-Paper.pdf).

This code compares these approaches on the 24 functions of the Black-Box Optimization Benchmarking (BBOB) suite from the [COCO](https://arxiv.org/pdf/1603.08785.pdf) benchmarking environment using their definition from [IOHprofiler](https://iohprofiler.github.io/). It is based on the original repositories and modules of the selected algorithms [vanilla Bayesian Optimization](https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html), [CMA-ES](https://github.com/CMA-ES/pycma), [random search](https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html), [SAASBO](https://github.com/martinjankowiak/saasbo), [RDUCB](https://github.com/huawei-noah/HEBO/tree/master/RDUCB), [PCA-BO](https://github.com/wangronin/Bayesian-Optimization/tree/KPCA-BO), [KPCA-BO](https://github.com/wangronin/Bayesian-Optimization/tree/KPCA-BO) and [TuRBO](https://github.com/uber-research/TuRBO). 

# Libraries and dependencies

The implementation is in Python 3.10.12 and all the libraries used are listed in `requirements.txt`.

# Structure
- `run_experiment.py` is the main file, used to run any experiments. It initializes the main setting of the experiment, calls the chosen algorithm, and writes log files. It takes as argument a file `.json` that is the output of the file `gen_config.py`.
- `wrapper.py` contains the definition of all algorithms and the method `wrapopt` that runs the main loop of the chosen algorithm. It is called by the file `run_experiment.py`.
- `my_logger.py` defines all the functions needed to generate the log files, storing the output data generated in a run. It is called by the file `run_experiment.py`.
- `total_config.json` allows the user to define the settings of an experiment. It is taken as an argument by the file `gen_config.py`. 
- `gen_config.py` generates a folder called `configs` containing files to run experiments based on the settings defined in `total_config.json`. 
- `mylib` stores the libraries with the implementation of the compared algorithms.
- `bayes_optim.zip` contains the bayes-optim package, with slight modifications to track the CPU time fofr CMA-ES.
- `Bayesian-Optimization.zip` contains the cloned repository [Bayesian-Optimization](https://github.com/wangronin/Bayesian-Optimization/tree/KPCA-BO) with some changes to track the CPU time for the algorithms PCA-BO and KPCA-BO.
- `RDUCB.zip` contains the cloned repository [RDUCB](https://github.com/huawei-noah/HEBO/tree/master/RDUCB) with modifications to track the CPU time for the algorithm RDUCB.
- `GPy.zip` and `GPyOpt.zip` contains the modules Gpy and GpyOpt, respectively, with modifications to track the CPU time for the algorithm RDUCB.
- `skopt.zip` contains the module skopt with some changes to track the CPU time for the algorithm vanilla Bayesian Optimization.
- `requirements.txt` contains the list of all the project’s dependencies with the specific version of each dependency.

# Execution from source
## Dependencies to run from source

Running this code from source requires Python 3.10.12, and the libraries given in `requirements.txt` (Warning: preferably use a virtual environment for this specific project, to avoid breaking the dependencies of your other projects). In Ubuntu, installing the dependencies can be done using the following command:

```
pip install -r requirements.txt
```

## Specific modules to copy for tracking the CPU time in the log file
To correctly track the CPU time, this code needs some modified modules and a modified cloned repositories. Follow the steps below:

1. Unzip the folders `bayes_optim.zip`, `skopt.zip`, `Bayesian-Optimization.zip`, `GPy.zip`, `GPyOpt.zip` and `RDUCB.zip`:
```
unzip bayes_optim.zip
unzip skopt.zip
unzip Bayesian-Optimization.zip
unzip GPy.zip
unzip GPyOpt.zip
unzip RDUCB.zip
```
2. Find the path of the used Python site-packages directory:
```
python -m site
```
3. Move `bayes_optim`, `skopt`, `GPy` and `GPyOPt` to the used Python site-packages directory:
```
mv bayes_optim <found_path_site_packages>
mv skopt <found_path_site_packages>
mv GPy <found_path_site_packages>
mv GPyOPt <found_path_site_packages>
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
- `reps` is the number of run repetitions with the same settings (optimizer, function id, instance id, etc.). Different repetitions differ only for the seed. Inside the folder containing the results, a `configs` folder will be generated containing `reps` .json files, one for each repetition. The seed number for repetitions starts from 0. 
- `lb` and `ub` are the lower bound and the upper bound of the design domain. In the paper, they are fixed as -5 and 5, respectively.
- `extra` contains extra text information to store in the result folder.
  
The `configs` folder will contain the file .json for all the possible combinations of settings present in all the lists configured in `total_config.json` repeated `reps` times (changing the number of seeds from 0 to `reps`-1). The name of the file .json describes the specific setting: Optimizer (Opt), function (F), instance (Id), Dimension (Dim), Repetition/number of seeds -1 (Rep), and number of the experiment (NumExp) (ex. Opt-turbo1_F-1_Id-0_Dim-10_Rep-0_NumExp-0.json).
### Execute repetitions in parallel using a cluster
If a job scheduling system for Linux clusters is available, the batch script can be edited inside the file `gen_config.py`. 
After choosing the parameters and editing the batch script, a folder called `run_current_date_and_time` containing folders with the result data and the `configs` folder will be generated using the following command:
```
python gen_config.py total_config.json
```
and the jobs can be launched by typing the last command line that will appear as screen output.
### Execute a single run
If a job scheduling system is not available or there is no need to submit a job because only a single run is asked the following steps can be done. Here, there is no need to adjust the settings to generate the batch script editing the file `gen_config.py`. Therefore, after choosing the parameters the folder called `run_current_date_and_time` containing the folders with the result data, and the `configs` folder will be generated using the following command:
```
python gen_config.py total_config.json
```
then, move to the folder `run_current_date_and_time` typing the last command line that will appear as screen output.
A single run with specific settings can be executed using the following command:
```
python ../run_experiment.py configs/settings_you_want_to_run.json
```
## Analysis from source
Reps-folders for each function indicated in `fiids` with the first part of the name stored in `folder` inside the file `total_config.json` will be generated inside the folder `run_current_date_and_time`. Each of them contains a folder `data_number_and_name_of_the_function` that stores a `.dat` file with all the results about the loss and the different CPU times tracked (the loss examined in the paper is stored under the name `best-so-far f(x)`).


