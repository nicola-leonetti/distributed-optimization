import os

import matplotlib.pyplot as plt
import numpy as np
import pickle
from disropt.problems import Problem

from logistic_regression_parameters import *

# initialize
with open(os.path.join(RESULTS_DIR, 'info.pkl'), 'rb') as inp:
    info = pickle.load(inp)
NN = info['N']
iters = info['iterations']
size = info['size']

# load agent data
ADMM_sequence = np.zeros((NN, iters, size))
gt_sequence = np.zeros((NN, iters, size))
local_function = {}
for i in range(NN):
    ADMM_sequence[i, :, :] = np.load(os.path.join(
        RESULTS_DIR, f"agent_{i}_seq_admm.npy"))
    gt_sequence[i, :, :] = np.load(os.path.join(
        RESULTS_DIR, f"agent_{i}_seq_gradtr.npy"))
    with open(os.path.join(RESULTS_DIR, f'agent_{i}_func.pkl'), 'rb') as inp:
        local_function[i] = pickle.load(inp)

# solve centralized problem
global_obj_func = 0
for i in range(NN):
    global_obj_func += local_function[i]

global_problem = Problem(global_obj_func)
problem_solution = global_problem.solve()
cost_centr = global_obj_func.eval(problem_solution)
problem_solution = problem_solution.flatten()

# compute cost errors
cost_err_admm = np.zeros((NN, iters))
cost_err_gradtr = np.zeros((NN, iters))

for i in range(NN):
    for t in range(iters):
        # first compute global function value at local point
        cost_ii_tt_admm = 0
        cost_ii_tt_gradtr = 0
        for j in range(NN):
            cost_ii_tt_admm += local_function[j].eval(
                ADMM_sequence[i, t, :][:, None])
            cost_ii_tt_gradtr += local_function[j].eval(
                gt_sequence[i, t, :][:, None])

        # then compute errors
        cost_err_admm[i, t] = abs(cost_ii_tt_admm - cost_centr)
        cost_err_gradtr[i, t] = abs(cost_ii_tt_gradtr - cost_centr)

# plot maximum cost error
plt.figure()
plt.title('Maximum cost error (among agents)')
plt.xlabel(r"iteration $k$")
plt.ylabel(
    r"$\max_{i} \: \left|(\sum_{j=1}^N f_j(x_i^k) - f^\star)/f^\star \right|$")
plt.semilogy(np.arange(iters), np.amax(cost_err_admm /
             cost_centr, axis=0), label='ADMM-Tracking Gradient')
plt.semilogy(np.arange(iters), np.amax(cost_err_gradtr /
             cost_centr, axis=0), label='Gradient Tracking')
plt.legend()

# plot maximum solution error
admm_solution_error = np.linalg.norm(
    ADMM_sequence - problem_solution[None, None, :], axis=2)
gt_solution_error = np.linalg.norm(
    gt_sequence - problem_solution[None, None, :], axis=2)

plt.figure()
plt.title('Maximum solution error (among agents)')
plt.xlabel(r"iteration $k$")
plt.ylabel(r"$\max_{i} \: \|x_i^k - x^\star\|$")
plt.semilogy(np.arange(iters), np.amax(admm_solution_error, axis=0),
             label='ADMM-Tracking Gradient')
plt.semilogy(np.arange(iters), np.amax(
    gt_solution_error, axis=0), label='Gradient Tracking')
plt.legend()

plt.show()
