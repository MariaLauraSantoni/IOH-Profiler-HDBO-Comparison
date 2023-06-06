# Comparison of High-Dimensional Bayesian Optimization Algorithms on BBOB
This repository contains the code associated with Comparison of High-Dimensional Bayesian Optimization Algorithms on BBOB.

It proposes a modular algorithm framework to make the implementation of several algorithms compared within the paper compatible
via [IOHprofiler](https://iohprofiler.github.io/), and appropriate code to store all the data obtained.
The selected algorithms are: 
- Vanilla Bayesian Optimization taken from the python module [scikit-learn](https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html).
- CMA-ES from the [pycma](https://github.com/CMA-ES/pycma) package.
- SAASBO algorithm from [High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces](https://arxiv.org/pdf/2103.00349.pdf).
- EBO introduced in [Batched Large-scale Bayesian Optimization in High-dimensional Spaces](https://arxiv.org/pdf/1706.01445.pdf).
- PCA-BO proposed in [High Dimensional Bayesian Optimization Assisted by Principal Component Analysis](https://arxiv.org/pdf/2007.00925.pdf).
- KPCA-BO introduced in [High Dimensional Bayesian Optimization with Kernel Principal Component Analysis](https://arxiv.org/pdf/2204.13753.pdf).
- TuRBO from [Scalable Global Optimization via Local Bayesian Optimization](https://proceedings.neurips.cc/paper/2019/file/6c990b7aca7bc7058f5e98ea909e924b-Paper.pdf).

This code compares these approaches on the 24 functions of the Black-Box Optimization Benchmarking (BBOB) suite from the [COCO](https://arxiv.org/pdf/1603.08785.pdf) benchmarking environment suite using their definition from [IOHprofiler](https://iohprofiler.github.io/). It is based on the original repositories and modules of the selected algorithms [Vanilla Bayesian Optimization](https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html), [CMA-ES](https://github.com/CMA-ES/pycma), [SAASBO](https://github.com/martinjankowiak/saasbo), [EBO](https://github.com/zi-w/Ensemble-Bayesian-Optimization), [PCA-BO](https://github.com/wangronin/Bayesian-Optimization/tree/KPCA-BO), [KPCA-BO](https://github.com/wangronin/Bayesian-Optimization/tree/KPCA-BO) and [TuRBO](https://github.com/uber-research/TuRBO). We provide all the python files to run the paper experiments and to store the result in data files.

# Libraries and dependencies

The implementation of all task and algorithms and the experiments are performed with Python 3.7.4 and all the libraries used are listed in requirements.txt.

# Structure
- run_experiment.py is the main file, used to run any experiments and initialises the main setting of the experiment, call the chosen algorithm and the write log files. It takes as argument a file json that is the output of the file gen_config.py.
- wrapper.py contains the definition of all algorithms and the method wrapopt that  runs the main loop of the chosen algorithm. It is called by run_experiment.py.
- my_logger.py defines all the functions needed to generate the files to store the data output by a run. It is called by run_experiment.py.
- total_config.json allows defining the settings of the experiment and it has to be the argument of the file gen_config.py 
- gen_config.py generates config file to run each algorithm with the parameters chosen in total_config.json given as an input.
- mylib contains one folder for each algorithm with all the classes and functions needed to run them.
- Bayesian-Optimization.zip
- sksparse.zip
- skopt.zip 
- requirements.txt


