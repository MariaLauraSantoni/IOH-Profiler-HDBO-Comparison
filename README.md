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
