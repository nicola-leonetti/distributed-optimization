import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pickle
from disropt.problems import Problem

RESULTS_DIR_BASE = "logistic_regression_results"

# Initialize numpy arrays to store average result sequences and their standard deviations
avg_cost_error_admm = np.zeros((1001))
avg_cost_error_gt = np.zeros((1001))
avg_cost_error_giant = np.zeros((1001))

std_cost_error_admm = np.zeros((1001))
std_cost_error_gt = np.zeros((1001))
std_cost_error_giant = np.zeros((1001))

avg_solution_error_admm = np.zeros((1001))
avg_solution_error_gt = np.zeros((1001))
avg_solution_error_giant = np.zeros((1001))

std_solution_error_admm = np.zeros((1001))
std_solution_error_gt = np.zeros((1001))
std_solution_error_giant = np.zeros((1001))

# Store all cost and solution errors to compute standard deviation later
all_cost_errors_admm = []
all_cost_errors_gt = []
all_cost_errors_giant = []

all_solution_errors_admm = []
all_solution_errors_gt = []
all_solution_errors_giant = []

number_of_simulations = 0
for d in os.listdir("."):
    if d.startswith(RESULTS_DIR_BASE + "_"):
        number_of_simulations += 1

for directory in filter(lambda d: d.startswith(
        RESULTS_DIR_BASE + "_"), os.listdir(".")):

    # Read number_of_agents, iterations, size and seed
    with open(os.path.join(directory, 'info.pkl'), 'rb') as inp:
        info = pickle.load(inp)
    number_of_agents = info['N']
    iterations = info['iterations'] + 1
    size = info['size']
    seed = int(directory.split("_")[-1])

    print(f"\nLoading data for simulation with seed {seed}...")

    # Read ADMM_sequence, gt_sequence, GIANT_sequence and local_function
    ADMM_sequence = np.zeros((number_of_agents, iterations, size))
    gt_sequence = np.zeros((number_of_agents, iterations, size))
    GIANT_sequence = np.zeros((number_of_agents, iterations, size))
    local_function = {}
    points = []
    labels = []
    for i in range(number_of_agents):
        # Load sequences
        ADMM_sequence[i, :, :] = np.load(os.path.join(
            directory, f"agent_{i}_seq_admm.npy"))
        gt_sequence[i, :, :] = np.load(os.path.join(
            directory, f"agent_{i}_seq_gradtr.npy"))
        GIANT_sequence[i, :, :] = np.load(os.path.join(
            directory, f"agent_{i}_seq_giant.npy"))

        # Load local function
        with open(os.path.join(directory, f'agent_{i}_func.pkl'), 'rb') as inp:
            local_function[i] = pickle.load(inp)

        # Load points and labels
        points.append(np.load(os.path.join(
            directory, f"agent_{i}_points.npy")))
        labels.append(np.load(os.path.join(
            directory, f"agent_{i}_labels.npy")))

    # Combinate points and labels of each agent together
    all_points = np.hstack(points)
    all_labels = np.vstack(labels).flatten()

    # Calculate global_obj_function
    global_obj_func = 0
    for i in range(number_of_agents):
        global_obj_func += local_function[i]

    # Suppress the warning about solving the problem with ECOS solver by
    # default
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Calculate problem_solution and cost_centr (value of objective
    # function calculated at the problem's solution)
    global_problem = Problem(global_obj_func)
    problem_solution = global_problem.solve()
    cost_centr = global_obj_func.eval(problem_solution)
    problem_solution = problem_solution.flatten()
    print(f"Actual solution: {problem_solution}")
    print(f"Gradient tracking solution: {gt_sequence[0][-1].flatten()}")
    print(f"ADMM-Tracking Gradient solution: {ADMM_sequence[0][-1].flatten()}")
    print(f"GIANT-ADMM solution: {GIANT_sequence[0][-1].flatten()}")
    print("Loaded")

    # Calculate cost_error_admm, cost_error_gt and cost_error_giant
    cost_error_admm = np.zeros((number_of_agents, iterations))
    cost_error_gt = np.zeros((number_of_agents, iterations))
    cost_error_giant = np.zeros((number_of_agents, iterations))
    # For each agent 0...n-1
    for i in range(number_of_agents):
        # For each iteration 0...1000
        for t in range(iterations):

            # Calculate function value at the estimated solution by
            # adding up the contributes of each agent
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

                # Calculate norm of the difference between function
                # value at estimated solution and actual problem's
                # solution
                cost_error_admm[i, t] = abs(
                    cost_ii_tt_admm.item() - cost_centr.item())
                cost_error_gt[i, t] = abs(
                    cost_ii_tt_gradtr.item() - cost_centr.item())
                cost_error_giant[i, t] = abs(
                    cost_ii_tt_giant.item() - cost_centr.item())

    # For each iteration, only consider the maximum cost error among all
    # agents
    cost_error_admm = np.amax(cost_error_admm, axis=0)
    cost_error_gt = np.amax(cost_error_gt, axis=0)
    cost_error_giant = np.amax(cost_error_giant, axis=0)

    # Store all cost errors
    all_cost_errors_admm.append(cost_error_admm)
    all_cost_errors_gt.append(cost_error_gt)
    all_cost_errors_giant.append(cost_error_giant)

    # Add everything to the respective global result
    avg_cost_error_admm += cost_error_admm
    avg_cost_error_gt += cost_error_gt
    avg_cost_error_giant += cost_error_giant

    # Calculate solution error
    solution_error_admm = np.linalg.norm(
        ADMM_sequence - problem_solution[None, None, :], axis=2)
    solution_error_gt = np.linalg.norm(
        gt_sequence - problem_solution[None, None, :], axis=2)
    solution_error_giant = np.linalg.norm(
        GIANT_sequence - problem_solution[None, None, :], axis=2)

    # For each iteration, only consider the maximum solution error among
    # all agents
    solution_error_admm = np.amax(solution_error_admm, axis=0)
    solution_error_gt = np.amax(solution_error_gt, axis=0)
    solution_error_giant = np.amax(solution_error_giant, axis=0)

    # Store all solution errors
    all_solution_errors_admm.append(solution_error_admm)
    all_solution_errors_gt.append(solution_error_gt)
    all_solution_errors_giant.append(solution_error_giant)

    # Add everything to the respective global result
    avg_solution_error_admm += solution_error_admm
    avg_solution_error_gt += solution_error_gt
    avg_solution_error_giant += solution_error_giant

    # Plot all the points, colored according to each point's label
    plt.figure(figsize=(8, 6))
    plt.title('Logistic Regression Decision Boundary and Data Points')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    # Plot all points of all agents
    plt.scatter(all_points[0, all_labels == 1], all_points[1,
                all_labels == 1], color='blue', marker='o', label='Class 1')
    plt.scatter(all_points[0, all_labels == -1], all_points[1,
                all_labels == -1], color='red', marker='x', label='Class -1')
    # Plot the decision boundary from the problem's solution
    x_vals = np.linspace(
        min(all_points[0, :]) - 1, max(all_points[0, :]) + 1, 200)
    y_vals = -(problem_solution[0] * x_vals +
               problem_solution[2]) / problem_solution[1]
    plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')
    plt.legend()
    plt.grid(True)

# Calculate the mean by dividing for the number of simulations
avg_cost_error_admm /= number_of_simulations
avg_cost_error_gt /= number_of_simulations
avg_cost_error_giant /= number_of_simulations
avg_solution_error_admm /= number_of_simulations
avg_solution_error_gt /= number_of_simulations
avg_solution_error_giant /= number_of_simulations

# Calculate the standard deviation
std_cost_error_admm = np.std(all_cost_errors_admm, axis=0)
std_cost_error_gt = np.std(all_cost_errors_gt, axis=0)
std_cost_error_giant = np.std(all_cost_errors_giant, axis=0)
std_solution_error_admm = np.std(all_solution_errors_admm, axis=0)
std_solution_error_gt = np.std(all_solution_errors_gt, axis=0)
std_solution_error_giant = np.std(all_solution_errors_giant, axis=0)

# Plot

# When plotting confidence bands, the standard deviation is multiplied
# this factor
confidence_band_scale = 1/3

# Maximum average cost error
plt.figure()
plt.title('Maximum average cost error (among agents)')
plt.xlabel(r"iteration $k$")
plt.ylabel(
    r"$\max_{i} \: \left|(\sum_{j=1}^N f_j(x_i^k) - f^\star)/f^\star \right|$")
plt.semilogy(np.arange(iterations), avg_cost_error_admm,
             label='ADMM-Tracking Gradient')
plt.fill_between(np.arange(iterations), avg_cost_error_admm - confidence_band_scale*std_cost_error_admm,
                 avg_cost_error_admm + confidence_band_scale*std_cost_error_admm, alpha=0.2)
plt.semilogy(np.arange(iterations), avg_cost_error_gt,
             label='Gradient Tracking')
plt.fill_between(np.arange(iterations), avg_cost_error_gt - confidence_band_scale*std_cost_error_gt,
                 avg_cost_error_gt + confidence_band_scale*std_cost_error_gt, alpha=0.2)
plt.semilogy(np.arange(iterations), avg_cost_error_giant, label='GIANT-ADMM')
plt.fill_between(np.arange(iterations), avg_cost_error_giant - confidence_band_scale*std_cost_error_giant,
                 avg_cost_error_giant + confidence_band_scale*std_cost_error_giant, alpha=0.2)
plt.legend()

# Maximum average solution error
plt.figure()
plt.title('Maximum average solution error (among agents)')
plt.xlabel(r"iteration $k$")
plt.ylabel(r"$\max_{i} \: \|x_i^k - x^\star\|$")
plt.semilogy(np.arange(iterations), avg_solution_error_admm,
             label='ADMM-Tracking Gradient')
plt.fill_between(np.arange(iterations), avg_solution_error_admm - confidence_band_scale*std_solution_error_admm,
                 avg_solution_error_admm + confidence_band_scale*std_solution_error_admm, alpha=0.2)
plt.semilogy(np.arange(iterations), avg_solution_error_gt,
             label='Gradient Tracking')
plt.fill_between(np.arange(iterations), avg_solution_error_gt - confidence_band_scale*std_solution_error_gt,
                 avg_solution_error_gt + confidence_band_scale*std_solution_error_gt, alpha=0.2)
plt.semilogy(np.arange(iterations),
             avg_solution_error_giant, label='GIANT-ADMM')
plt.fill_between(np.arange(iterations), avg_solution_error_giant - confidence_band_scale*std_solution_error_giant,
                 avg_solution_error_giant + confidence_band_scale*std_solution_error_giant, alpha=0.2)
plt.legend()


plt.show()
