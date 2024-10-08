import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pickle
from disropt.problems import Problem

from logistic_regression_parameters import *

# Load number of agents, iterations and variable size from info.pkl
with open(os.path.join(RESULTS_DIR, 'info.pkl'), 'rb') as inp:
    info = pickle.load(inp)
number_of_agents = info['N']
iters = info['iterations'] + 1
size = info['size']

# load agent data
ADMM_sequence = np.zeros((number_of_agents, iters, size))
gt_sequence = np.zeros((number_of_agents, iters, size))
GIANT_sequence = np.zeros((number_of_agents, iters, size))
local_function = {}
for i in range(number_of_agents):
    ADMM_sequence[i, :, :] = np.load(os.path.join(
        RESULTS_DIR, f"agent_{i}_seq_admm.npy"))
    gt_sequence[i, :, :] = np.load(os.path.join(
        RESULTS_DIR, f"agent_{i}_seq_gradtr.npy"))
    GIANT_sequence[i, :, :] = np.load(os.path.join(
        RESULTS_DIR, f"agent_{i}_seq_giant.npy"))
    with open(os.path.join(RESULTS_DIR, f'agent_{i}_func.pkl'), 'rb') as inp:
        local_function[i] = pickle.load(inp)

# solve centralized problem
global_obj_func = 0
for i in range(number_of_agents):
    global_obj_func += local_function[i]

# Suppress the warning about solving the problem with ECOS solver by
# default
warnings.simplefilter(action='ignore', category=FutureWarning)

global_problem = Problem(global_obj_func)
problem_solution = global_problem.solve()
cost_centr = global_obj_func.eval(problem_solution)
problem_solution = problem_solution.flatten()

print(f"Actual solution: {problem_solution}")
print(f"Gradient tracking solution: {gt_sequence[0][-1].flatten()}")
print(f"ADMM-Tracking Gradient solution: {ADMM_sequence[0][-1].flatten()}")
print(f"GIANT-ADMM solution: {GIANT_sequence[0][-1].flatten()}")

# compute cost errors
cost_err_admm = np.zeros((number_of_agents, iters))
cost_err_gradtr = np.zeros((number_of_agents, iters))
cost_err_giant = np.zeros((number_of_agents, iters))

for i in range(number_of_agents):
    for t in range(iters):
        # first compute global function value at local point
        cost_ii_tt_admm = 0
        cost_ii_tt_gradtr = 0
        cost_ii_tt_giant = 0
        for j in range(number_of_agents):
            cost_ii_tt_admm += local_function[j].eval(
                ADMM_sequence[i, t, :][:, None])
            cost_ii_tt_gradtr += local_function[j].eval(
                gt_sequence[i, t, :][:, None])
            cost_ii_tt_giant += local_function[j].eval(
                GIANT_sequence[i, t, :][:, None])

        # then compute errors
        cost_err_admm[i, t] = abs(cost_ii_tt_admm.item() - cost_centr.item())
        cost_err_gradtr[i, t] = abs(
            cost_ii_tt_gradtr.item() - cost_centr.item())
        cost_err_giant[i, t] = abs(cost_ii_tt_giant.item() - cost_centr.item())

# compute maximum consensus error
avg_admm_sequence = np.mean(ADMM_sequence, axis=0)
avg_gt_sequence = np.mean(gt_sequence, axis=0)
avg_giant_sequence = np.mean(GIANT_sequence, axis=0)

# plot maximum cost error
plt.figure()
plt.title('Maximum cost error (among agents)')
plt.xlabel(r"iteration $t$")
plt.ylabel(
    r"$\max_{i} \: \left|(\sum_{j=1}^N f_j(x_i^t) - f^\star)/f^\star \right|$")
plt.semilogy(np.arange(iters), np.amax(cost_err_admm /
             cost_centr, axis=0), label='ADMM-Tracking Gradient')
plt.semilogy(np.arange(iters), np.amax(cost_err_gradtr /
             cost_centr, axis=0), label='Gradient Tracking')
plt.semilogy(np.arange(iters), np.amax(cost_err_giant /
                                       cost_centr, axis=0), label='GIANT-ADMM')
plt.legend()

# plot maximum solution error
admm_solution_error = np.linalg.norm(
    ADMM_sequence - problem_solution[None, None, :], axis=2)
gt_solution_error = np.linalg.norm(
    gt_sequence - problem_solution[None, None, :], axis=2)
giant_solution_error = np.linalg.norm(
    GIANT_sequence - problem_solution[None, None, :], axis=2)

plt.figure()
plt.title('Maximum solution error (among agents)')
plt.xlabel(r"iteration $t$")
plt.ylabel(r"$\max_{i} \: \|x_i^t - x^\star\|$")
plt.semilogy(np.arange(iters), np.amax(admm_solution_error, axis=0),
             label='ADMM-Tracking Gradient')
plt.semilogy(np.arange(iters), np.amax(
    gt_solution_error, axis=0), label='Gradient Tracking')
plt.semilogy(np.arange(iters), np.amax(
    giant_solution_error, axis=0), label='GIANT-ADMM')
plt.legend()

plt.show()
