#!/usr/bin/env python
# coding: utf-8

# # Introduction
# As described in the repo README, the experiment runner provides the fixture for running experiments on UCI data sets. One can also use this repo as a module providing an API to the RPA-GP family of GP models. 
# 
# Below are demonstrations on how to create and use these models on example data.

# In[15]:


# Preliminary imports
import torch
from torch.optim import Adam
from gpytorch.kernels import RBFKernel, ScaleKernel, AdditiveStructureKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood


# In[16]:


from gp_experiment_runner import _determine_folds, _access_fold, _normalize_by_train


# In[17]:


# Load a data set
from gp_experiment_runner import load_dataset  # assumes UCI datasets are available locally
dataset = load_dataset('yacht') # returns a Pandas dataframe

n_train = len(dataset) // 10 * 9
train = dataset.iloc[:n_train]
test = dataset.iloc[n_train:]

# Or use utilities from repo:
# from gp_experiment_runner import _determine_folds, _access_fold, _normalize_by_train
# folds = _determine_folds(0.1, dataset)
# train, test = _access_fold(dataset, folds, 0)
# train, test = _normalize_by_train(train, test)

train_x = torch.tensor(train.iloc[:, :-1].values, dtype=torch.float)
train_y = torch.tensor(train.iloc[:, -1].values, dtype=torch.float)
test_x = torch.tensor(test.iloc[:, :-1].values, dtype=torch.float)
test_y = torch.tensor(test.iloc[:, -1].values, dtype=torch.float)


# ### Create an RPA-GP model

# In[4]:


from gp_models import ExactGPModel, ScaledProjectionKernel
import rp
n, d = train_x.shape
num_projs = 20

# Draw random projections and store in a linear module
# Here, we are drawing 20 Gaussian projections into 1 dimension.
projs = [rp.gen_rp(d, 1, dist='gaussian') for _ in range(num_projs)]
proj_module = torch.nn.Linear(d, num_projs, bias=False)
proj_module.weight.data = torch.cat(projs, dim=1).t()

# Create the additive model that operates over these projections
# Fixing the outputscale and lengthscale of the base kernels.
base_kernel = RBFKernel()
base_kernel.initialize(lengthscale=torch.tensor([1.]))
base_kernel = ScaleKernel(base_kernel)
base_kernel.initialize(outputscale=torch.tensor([1/num_projs]))

# Combine into a single module.
kernel = ScaledProjectionKernel(proj_module, base_kernel, 
                                prescale=False,
                                ard_num_dims=num_projs,
                                learn_proj=False)
# Or, just call the method from training_routines that wraps this initialization
# from training_routines import create_additive_rp_kernel
# create_additive_rp_kernel(d, num_projs, learn_proj=False, kernel_type='RBF', 
#                           space_proj=False, prescale=False, ard=True, k=1, 
#                           proj_dist='gaussian')

kernel = ScaleKernel(kernel) # Optionally wrap with an additional ScaleKernel 

# Create an ExactGP model with this kernel
likelihood = GaussianLikelihood()
likelihood.noise = 1.
model = ExactGPModel(train_x, train_y, likelihood, kernel)


# ### Train the model

# In[5]:


# Train the model
mll = ExactMarginalLogLikelihood(model.likelihood, model)
mll.train()
optimizer = Adam(mll.parameters(), lr=0.01)
for iteration in range(1000):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()
    msg = "Iter {}: Loss={:2.4f}, Noise={:2.4f}".format(iteration, loss.item(), model.likelihood.noise.item())
    print(msg)


# In[6]:


# Test the model
model.eval()
output = model(test_x)
rmse = (test_y - output.mean).pow(2).mean().sqrt().item()
neg_log_lik = -mll(output, test_y).item()
print('RMSE={:2.4f}, Neg Log Likelihood={:2.4f}'.format(rmse, neg_log_lik))


# In[ ]:

#
#
#
#
# # ### Create an DPA-GP-ARD model
#
# # In[7]:
#
#
# # Draw random projections, `diversify` and store in a linear module
# # Here, we are drawing 20 Gaussian projections into 1 dimension.
# projs = [rp.gen_rp(d, 1, dist='gaussian') for _ in range(num_projs)] # initial directions
#
# newW, _ = rp.space_equally(torch.cat(projs,dim=1).t(), lr=0.1, niter=5000) # Try to diversify
# newW.requires_grad = False # Make sure they aren't trainable
# projs = [newW[i:i+1, :].t() for i in range(0, num_projs, 1)]  # Reshape like initial directions
#
# proj_module = torch.nn.Linear(d, num_projs, bias=False)
# proj_module.weight.data = torch.cat(projs, dim=1).t()
#
# # Create the additive model that operates over these projections
# # Fixing the outputscale and lengthscale of the base kernels.
# base_kernel = RBFKernel()
# base_kernel.initialize(lengthscale=torch.tensor([1.]))
# base_kernel = ScaleKernel(base_kernel)
# base_kernel.initialize(outputscale=torch.tensor([1/num_projs]))
#
# # Combine into a single module.
# # Using prescale=True applies lengthscales to original input space, i.e. scaling pre-projection.
# kernel = ScaledProjectionKernel(proj_module, base_kernel,
#                                 prescale=True,
#                                 ard_num_dims=d,
#                                 learn_proj=False)
# # Or, just call the method from training_routines that wraps this initialization
# # from training_routines import create_additive_rp_kernel
# # create_additive_rp_kernel(d, num_projs, learn_proj=False, kernel_type='RBF',
# #                           space_proj=True, prescale=True, ard=True, k=1,
# #                           proj_dist='gaussian')
#
# kernel = ScaleKernel(kernel) # Optionally wrap with an additional ScaleKernel
# 
# # Create an ExactGP model with this kernel
# likelihood = GaussianLikelihood()
# likelihood.noise = 1.
# model = ExactGPModel(train_x, train_y, GaussianLikelihood(), kernel)
#
#
# # ### Train the model
#
# # In[8]:
#
#
# # Train the model
# mll = ExactMarginalLogLikelihood(model.likelihood, model)
# model.train()
# model.likelihood.train()
# optimizer = Adam(mll.parameters(), lr=0.01)
# for iteration in range(1000):
#     optimizer.zero_grad()
#     output = model(train_x)
#     loss = -mll(output, train_y)
#     loss.backward()
#     optimizer.step()
#     msg = "Iter {}: Loss={:2.4f}, Noise={:2.4f}".format(iteration, loss.item(), model.likelihood.noise.item())
#     print(msg)
#
#
# # In[9]:
#
#
# # Test the model
# with torch.no_grad():
#     model.eval()
#     model.likelihood.eval()
#     output = model(test_x)
#     rmse = (test_y - output.mean).pow(2).mean().sqrt().item()
#     neg_log_lik = -mll(output, test_y).item()
#     print('RMSE={:2.4f}, Neg Log Likelihood={:2.4f}'.format(rmse, neg_log_lik))
#
#
# # In[ ]:




