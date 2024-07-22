# TODO Confronto la norma della differenza tra x corrente e  star
# con scala logaritmica su asse y
# TODO Faccio tuning dei parametri
# TODO Studiare logistic regression da paper
# TODO (parla con Carnevale)
# TODO Confronta gli algoritmi per capire come si comporano con un
# linear classifier (logistic regression)

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import os

from disropt.problems import Problem

N = np.load("agents.npy")  # number of agents
d = 2  # variable size

# Load global function from files
global_function = 0
for i in range(N):
    with open(f"agent_{i}_function.pkl", "rb") as f:
        global_function += pickle.load(f)

# Load sequences from the respective results directories
gt_sequence = {i: np.load(os.path.join("gt_results",
                                       f"agent_{i}_sequence.npy"))
               for i in range(N)}
admm_sequence = {i: np.load(os.path.join("admm_results",
                                         f"agent_{i}_sequence.npy"))
                 for i in range(N)}
giant_admm_sequence = {i: np.load(os.path.join("giant_admm_results",
                                               f"agent_{i}_sequence.npy"))
                       for i in range(N)}


actual_solution = Problem(global_function).solve()
print("Simulation results:")
print(f"Actual solution: {actual_solution.flatten()}")
print(f"Gradient tracking: {gt_sequence[0][-1].flatten()}")
print(f"ADMM: {admm_sequence[0][-1].flatten()}")
print(f"GIANT-ADMM: {giant_admm_sequence[0][-1].flatten()}")


def get_loss_sequence(sequence):
    """
    For each sequence, it returns a sequence of elements 
    [iteration, norm(estimate[iteration] - actual_solution)]
    """
    loss_sequence = {}
    norm = np.linalg.norm
    


norm = np.linalg.norm
iterations = gt_sequence[0].shape[0]
# for each agent
for i in range(N):
    gt_loss_sequence[i] = [norm(gt_sequence[i][iteration - 1, :, :] - actual_solution)
                           for iteration in range(iterations)]

titles = [
    "Gradient tracking",
    "ADMM-tracking gradient",
    "GIANT-ADMM",
]

for p, sequence in enumerate([gt_sequence, admm_sequence, giant_admm_sequence]):
    colors = {i: np.random.rand(3, 1).flatten() for i in range(N)}

    fig, axs = plt.subplots(2)
    fig.suptitle(titles[p])

    # Per ogni agente
    for i in range(N):
        dims = sequence[i].shape
        iterations = dims[0]

        # Plot evolution of local estimates compared to actual solution
        # Per ognuna delle dimensioni della x
        for j in range(dims[1]):
            axs[0].plot(np.arange(iterations),
                        sequence[i][:, j, 0], color=colors[i])
            # Plot actual solution as a dotted line
            axs[0].axhline(y=actual_solution[j], linestyle='--')

        # plt.legend([f"Agent {i}" for i in range(N)])

        # Plot loss
        for j in range(dims[1]):
            axs[1].plot(np.arange(iterations),
                        gt_loss_sequence[i][:], color=colors[i])
            axs[1].set_yscale("log")


plt.show()
