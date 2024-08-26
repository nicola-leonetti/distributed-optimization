'''
Solution of a cost-coupled problem: logistic regression for
classification.

Each agent has a certain number of randomly generated points, each 
labeled 1 or -1. The points are generated by agents according to a 
multivariate normal distribution, with different mean and covariance for
the two labels.

The results are obtained using the following algorithms:
- Gradient Tracking
- ADMM-Tracking Gradient
- GIANT-ADMM

The results are saved in some files for further analysis.
'''

import dill as pickle
import numpy as np
import os
from mpi4py import MPI
from disropt.agents import Agent
from disropt.algorithms import SubgradientMethod, GradientTracking, DualDecomposition
from disropt.functions import Variable, SquaredNorm, Logistic
from disropt.utils.graph_constructor import binomial_random_graph, metropolis_hastings
from disropt.problems import Problem

from admm_tracking_gradient import ADMMTrackingGradient
from giant_admm import GIANTADMM
from logistic_regression_parameters import *


number_of_agents = MPI.COMM_WORLD.Get_size()
agent_id = MPI.COMM_WORLD.Get_rank()

# Generate a random graph, common to all agents, and the corresponding
# weights matrix
graph = binomial_random_graph(number_of_agents, p=0.3, seed=1)
W = metropolis_hastings(graph)
# Initialize a different RNG for each agent
np.random.seed(10*agent_id)

# Gaussian parameters
mu = (np.array([0, 0]).transpose(), np.array([3, 2]).transpose())
sigma = (np.eye(2), np.eye(2))

sample_space_dimension = mu[0].shape[0]

number_of_samples_range = (np.random.randint(2, 6), np.random.randint(2, 6))

# Generate the agent's points
# For each of those, a multivariate (2D) normal distribution is
# used, meaning that each of those points will be generated in a 2D
# space around a certain mean point. In this way, two clusters of point
# will be formed
points = np.zeros((sample_space_dimension,
                  number_of_samples_range[0] + number_of_samples_range[1]))
points[:, 0:number_of_samples_range[0]] = np.random.multivariate_normal(
    mu[0], sigma[0], number_of_samples_range[0]).transpose()
points[:, number_of_samples_range[0]:] = np.random.multivariate_normal(
    mu[1], sigma[1], number_of_samples_range[1]).transpose()

# Label each point 1 or -1 randomly
labels = np.ones((sum(number_of_samples_range), 1))
labels[number_of_samples_range[0]:] = -labels[number_of_samples_range[0]:]

# Initialize the problem's cost function, based on the agent's points
z = Variable(sample_space_dimension+1)
A = np.ones((sample_space_dimension+1, 1))
A[-1] = 0
obj_func = (C / (2 * number_of_agents)) * SquaredNorm(A @ z)
for j in range(sum(number_of_samples_range)):
    e_j = np.zeros((sum(number_of_samples_range), 1))
    e_j[j] = 1
    A_j = np.vstack((points @ e_j, 1))
    obj_func += Logistic(- labels[j] * A_j @ z)

agent = Agent(
    in_neighbors=np.nonzero(graph[agent_id, :])[0].tolist(),
    out_neighbors=np.nonzero(graph[:, agent_id])[0].tolist(),
    in_weights=W[agent_id, :].tolist()
)
problem = Problem(obj_func)
agent.set_problem(problem)

# Initialize the agents' initial conditions with random values
# In order to make the simulation fair, each agent should start with the
# same random value as its initial condition
np.random.seed(sample_space_dimension)
x0 = 5*np.random.rand(sample_space_dimension+1, 1)

gradient_tracking = GradientTracking(
    agent=agent,
    initial_condition=x0,
    enable_log=True
)
initial_z = {i: 10*np.random.rand(2*(sample_space_dimension+1), 1)
             for i in agent.in_neighbors}
ADMM_tracking_gradient = ADMMTrackingGradient(
    agent=agent,
    initial_condition=x0,
    initial_z=initial_z,
    gamma=ADMM_gamma,
    rho=ADMM_rho,
    alpha=ADMM_alpha,
    enable_log=True
)
GIANT_ADMM = GIANTADMM(
    agent=agent,
    initial_condition=x0,
    initial_z=initial_z,
    gamma=GIANT_gamma,
    rho=GIANT_rho,
    alpha=GIANT_alpha,
    enable_log=True
)

# Run the algorithms
gt_sequence = gradient_tracking.run(
    iterations=iterations, stepsize=GT_stepsize)
ADMM_sequence = ADMM_tracking_gradient.run(
    iterations=iterations, stepsize=ADMM_stepsize)
GIANT_sequence = GIANT_ADMM.run(iterations=iterations, stepsize=GIANT_stepsize)

# Insert initial condition in the sequences
gt_sequence = np.insert(gt_sequence, 0, x0, axis=0)
ADMM_sequence = np.insert(ADMM_sequence, 0, x0, axis=0)
GIANT_sequence = np.insert(GIANT_sequence, 0, x0, axis=0)

print(f"Gradient tracking: agent {agent_id}: {
      gradient_tracking.get_result().flatten()}")
print(f"ADMM-Tracking Gradient: agent {agent_id}: {
    ADMM_tracking_gradient.get_result().flatten()}")
print(f"GIANT-ADMM: agent {agent_id}: {GIANT_ADMM.get_result().flatten()}")

# Save:
# - the number of agents, the size of the space and the number of
#   iterations in "info.pkl"
# - the function of the i-th agent in "agent_i_func.pkl"
# - solution sequences for the various algorithms
if agent_id == 0:
    with open(os.path.join(RESULTS_DIR, 'info.pkl'), 'wb') as output:
        pickle.dump({'N': number_of_agents, 'size': sample_space_dimension+1, 'iterations': iterations},
                    output, pickle.HIGHEST_PROTOCOL)

with open(os.path.join(RESULTS_DIR, f'agent_{agent_id}_func.pkl'), 'wb') as output:
    pickle.dump(obj_func, output, pickle.HIGHEST_PROTOCOL)

np.save(os.path.join(RESULTS_DIR, f"agent_{
        agent_id}_seq_gradtr.npy"), np.squeeze(gt_sequence))
np.save(os.path.join(RESULTS_DIR, f"agent_{
        agent_id}_seq_admm.npy"), np.squeeze(ADMM_sequence))
np.save(os.path.join(RESULTS_DIR, f"agent_{
        agent_id}_seq_giant.npy"), np.squeeze(GIANT_sequence))

print(f"Agent {agent_id}: finished ")
