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
- `bayes_optim.zip` contains the bayes-optim package, with slight modifications to track the CPU time for CMA-ES.
- `Bayesian-Optimization.zip` contains the cloned repository [Bayesian-Optimization](https://github.com/wangronin/Bayesian-Optimization/tree/KPCA-BO) with some changes to track the CPU time for the algorithms PCA-BO and KPCA-BO.
- `RDUCB.zip` contains the cloned repository [RDUCB](https://github.com/huawei-noah/HEBO/tree/master/RDUCB) with modifications to track the CPU time for the algorithm RDUCB.
- `GPy.zip` and `GPyOpt.zip` contain the modules Gpy and GpyOpt, respectively, with modifications to track the CPU time for the algorithm RDUCB.
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
First of all, use the file `total_config.json` to decide the settings of your experiment: 
- `folder` is the prefix of the folders that are generated to store the results from the experiment. 
- `optimizers` is a list of as many strings as the number of algorithms that will be tested in the experiment. Possible names for the algorithms are `BO_sklearn`, `pyCMA`, `random`, `saasbo`, `RDUCB`, `linearPCABO`, `KPCABO`, `turbo1` and `turbom`.
- `fiids` is a list of functions to be optimized. The list can contain a single number or multiple integers identifying the 24 BBOB functions.
- `iids` is a list of problem instances. In our paper, 0, 1, and 2 are considered.
- `dims` is a list of problem dimensions tested within the experiment.
- `reps` is the number of run repetitions that will be performed under the same settings (optimizer, function id, instance id, etc.). Different repetitions differ only for different seeds. The seed number for repetitions starts from 0. 
- `lb` and `ub` are the lower and the upper bounds of the search space along each dimension. In the paper, they are fixed as -5 and 5, respectively.
- `extra` contains extra text information to store in the result folder.

Results will be generated inside a `run_[current_date_and_time]` folder. This contains a `configs` subfolder storing as many .json files as the total number of different settings defined by all the combinations of parameters in `total_config.json`. The name of each .json file describes the specific setting: optimizer (Opt), function (F), instance (Id), dimension (Dim), repetition (Rep), and a numerical experiment identifier utilized to denote the tested settings in ascending order (NumExp). For example, `Opt-turbo1_F-1_Id-0_Dim-10_Rep-0_NumExp-0.json`. 


### Execute parallel jobs using a cluster
If a job scheduling system for Linux clusters is available, the batch script to be edited is inside `gen_config.py`. Choosing the parameters in this file and script editing must be done before launching any jobs.

Use the command
```
python gen_config.py total_config.json
```
to generate a folder `run_[current_date_and_time]` containing the `configs` subfolder described above. Moreover, the same command generates an output to screen. Copy-paste the printed line (for example, `cd [path-to-root-folder]/run_15-04-2024_16h14m59s && for (( i=0; i<1; ++i )); do sbatch slurm$i.sh; done`) in your terminal window to start as many jobs as the setting combinations specified in `total_config.json`. At this point, the folders containing the results are generated inside `run_[current_date_and_time]`.

### Execute single runs
If a job scheduling system is not available or not necessary, the following steps must be followed. In this case, there is no need to adjust the settings to generate the batch script by editing the file `gen_config.py`. Again, after defining the experimental setup in `total_config.json`, run the command 
```
python gen_config.py total_config.json
```
to generate the folder `run_[current_date_and_time]` containing the `configs` subfolder described above. The command also prints to screen the experiment root folder `run_[current_date_and_time]` and how many files were generated, i.e., how many different settings are considered.
A single run for a specific setting can be started using the following command:
```
python run_experiment.py run_[current_date_and_time]/configs/[setting_you_want_to_run].json
```
and the folder containing the results is generated inside `run_[current_date_and_time]`.

### Running the code with Docker
To simplify the process of setting up and running the application, you can use Docker. The provided Dockerfile will create a consistent environment, ensuring that the application runs smoothly regardless of the host system.
Follow the steps below:

1. Install Docker on your system.
2. Build the Docker Image and run the Docker Container executing the following commands:
```
docker build --network=host . -t hdbo-docker
docker run -it hdbo-docker bash
```
3. After this, perform the normal commands to run a single run:
```
python gen_config.py total_config.json
python run_experiment.py run_[current_date_and_time]/configs/[setting_you_want_to_run].json
```
## Analysis from source
Each of the folders generated inside `run_[current_date_and_time]` contains a subfolder `data_number_and_name_of_the_function` that stores a `.dat` file generated by the logger. It tracks the loss evolution (under the name `best-so-far f(x)`) and the CPU times for a specific run. These are the metrics used in our paper to compare the performance of different algorithms. 
