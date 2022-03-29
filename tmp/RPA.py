"""
==================================
Bayesian optimization with `skopt`
==================================

Gilles Louppe, Manoj Kumar July 2016.
Reformatted by Holger Nahrstaedt 2020

.. currentmodule:: skopt

Problem statement
-----------------

We are interested in solving

.. math::
    x^* = arg \min_x f(x)

under the constraints that

- :math:`f` is a black box for which no closed form is known
  (nor its gradients);
- :math:`f` is expensive to evaluate;
- and evaluations of :math:`y = f(x)` may be noisy.

**Disclaimer.** If you do not have these constraints, then there
is certainly a better optimization algorithm than Bayesian optimization.

This example uses :class:`plots.plot_gaussian_process` which is available
since version 0.8.

Bayesian optimization loop
--------------------------

For :math:`t=1:T`:

1. Given observations :math:`(x_i, y_i=f(x_i))` for :math:`i=1:t`, build a
   probabilistic model for the objective :math:`f`. Integrate out all
   possible true functions, using Gaussian process regression.

2. optimize a cheap acquisition/utility function :math:`u` based on the
   posterior distribution for sampling the next point.
   :math:`x_{t+1} = arg \\min_x u(x)`
   Exploit uncertainty to balance exploration against exploitation.

3. Sample the next observation :math:`y_{t+1}` at :math:`x_{t+1}`.


Acquisition functions
---------------------

Acquisition functions :math:`u(x)` specify which sample :math:`x`: should be
tried next:

- Expected improvement (default):
  :math:`-EI(x) = -\\mathbb{E} [f(x) - f(x_t^+)]`
- Lower confidence bound: :math:`LCB(x) = \mu_{GP}(x) + \kappa \sigma_{GP}(x)`
- Probability of improvement: :math:`-PI(x) = -P(f(x) \geq f(x_t^+) + \kappa)`

where :math:`x_t^+` is the best point observed so far.

In most cases, acquisition functions provide knobs (e.g., :math:`\kappa`) for
controlling the exploration-exploitation trade-off.
- Search in regions where :math:`\mu_{GP}(x)` is high (exploitation)
- Probe regions where uncertainty :math:`\sigma_{GP}(x)` is high (exploration)
"""
from sklearn.datasets import make_regression

print(__doc__)

import numpy as np
np.random.seed(237)
import matplotlib.pyplot as plt
from skopt.plots import plot_gaussian_process



# import torch
# from torch.optim import Adam
# from gpytorch.kernels import RBFKernel, ScaleKernel, AdditiveStructureKernel
# from gpytorch.likelihoods import GaussianLikelihood
# from gpytorch.mlls import ExactMarginalLogLikelihood
# from gp_experiment_runner import _determine_folds, _access_fold, _normalize_by_train
# from gp_experiment_runner import load_dataset
# from gp_models import ExactGPModel, ScaledProjectionKernel
# import rp


from ioh import get_problem, logger

#############################################################################
# Toy example
# -----------
#
# Let assume the following noisy function :math:`f`:

noise_level = 0.1

dim = 10

#budget= 10 * dim + 50
#DoE_samples = int(.20 * budget)
budget=30
DoE_samples=10
# Common settings

ub = 5
lb = -5
f = get_problem(7, dimension=dim, instance=1, problem_type = 'Real')

# X, y = make_regression(n_samples=DoE_samples, n_features=dim, noise=100, random_state=0)
# print(X)
# print(y)





#############################################################################
# **Note.** In `skopt`, functions :math:`f` are assumed to take as input a 1D
# vector :math:`x`: represented as an array-like and to return a scalar
# :math:`f(x)`:.

# Plot f(x) + contours
# x = np.linspace(lb, ub, DoE_samples).reshape(-1, 1)*np.ones(dim)
# fx = [f(x_i) for x_i in x] #should be a list ok
# print(fx)
# plt.plot(x, fx, "r--", label="True (unknown)")
# plt.fill(np.concatenate([x, x[::-1]]),
#          np.concatenate(([fx_i - 1.9600 * noise_level for fx_i in fx],
#                           [fx_i + 1.9600 * noise_level for fx_i in fx[::-1]])),
#           alpha=.2, fc="r", ec="None")
# plt.legend()
# plt.grid()
# plt.show()

#############################################################################
# Bayesian optimization based on gaussian process regression is implemented in
# :class:`gp_minimize` and can be carried out as follows:

from skopt import gp_minimize

res = gp_minimize(f,                  # the function to minimize
                  list((((lb, ub), ) * dim )), # the bounds on each dimension of x
                  base_estimator='RPA',
                  acq_func="EI",      # the acquisition function
                  n_calls=budget,         # the number of evaluations of f
                  n_random_starts=DoE_samples,  # the number of random initialization points
                  noise=0.1**2,       # the noise level (optional)
                  random_state=1234)   # the random seed

#############################################################################
# Accordingly, the approximated minimum is found to be:

"x^*=%.4f, f(x^*)=%.4f" % (res.x[0], res.fun)

#############################################################################
# For further inspection of the results, attributes of the `res` named tuple
# provide the following information:
#
# - `x` [float]: location of the minimum.
# - `fun` [float]: function value at the minimum.
# - `models`: surrogate models used for each iteration.
# - `x_iters` [array]:
#   location of function evaluation for each iteration.
# - `func_vals` [array]: function value for each iteration.
# - `space` [Space]: the optimization space.
# - `specs` [dict]: parameters passed to the function.

print(res)

#############################################################################
# Together these attributes can be used to visually inspect the results of the
# minimization, such as the convergence trace or the acquisition function at
# the last iteration:
#
# from skopt.plots import plot_convergence
# plot_convergence(res);
#
# #############################################################################
# # Let us now visually examine
# #
# # 1. The approximation of the fit gp model to the original function.
# # 2. The acquisition values that determine the next point to be queried.

# plt.rcParams["figure.figsize"] = (8, 14)


# def f_wo_noise(x):
#     return f(x, noise_level=0)

#############################################################################
# Plot the 5 iterations following the 5 random points

# for n_iter in range(5):
#     # Plot true function.
#     plt.subplot(5, 2, 2*n_iter+1)
#
#     if n_iter == 0:
#         show_legend = True
#     else:
#         show_legend = False
#
#     ax = plot_gaussian_process(res, n_calls=n_iter,
#                                objective=f_wo_noise,
#                                noise_level=noise_level,
#                                show_legend=show_legend, show_title=False,
#                                show_next_point=False, show_acq_func=False)
#     ax.set_ylabel("")
#     ax.set_xlabel("")
#     # Plot EI(x)
#     plt.subplot(5, 2, 2*n_iter+2)
#     ax = plot_gaussian_process(res, n_calls=n_iter,
#                                show_legend=show_legend, show_title=False,
#                                show_mu=False, show_acq_func=True,
#                                show_observations=False,
#                                show_next_point=True)
#     ax.set_ylabel("")
#     ax.set_xlabel("")
#
# plt.show()

#############################################################################
# The first column shows the following:
#
# 1. The true function.
# 2. The approximation to the original function by the gaussian process model
# 3. How sure the GP is about the function.
#
# The second column shows the acquisition function values after every
# surrogate model is fit. It is possible that we do not choose the global
# minimum but a local minimum depending on the minimizer used to minimize
# the acquisition function.
#
# At the points closer to the points previously evaluated at, the variance
# dips to zero.
#
# Finally, as we increase the number of points, the GP model approaches
# the actual function. The final few points are clustered around the minimum
# because the GP does not gain anything more by further exploration:

# plt.rcParams["figure.figsize"] = (6, 4)

# Plot f(x) + contours
# _ = plot_gaussian_process(res, objective=f_wo_noise,
#                           noise_level=noise_level)
#
# plt.show()
