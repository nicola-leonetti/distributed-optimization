import dill as pickle
import numpy as np
import os
import time

from mpi4py import MPI
from disropt.agents import Agent
from disropt.algorithms.gradient_tracking import GradientTracking
from disropt.functions import QuadraticForm, Variable
from disropt.utils.utilities import is_pos_def
from disropt.utils.graph_constructor import binomial_random_graph, metropolis_hastings
from disropt.problems.problem import Problem

import parameters

RESULTS_DIR = "gt_results"

# get MPI info
comm = MPI.COMM_WORLD
nproc = comm.Get_size()
local_rank = comm.Get_rank()

# Generate a common graph (everyone use the same seed)
graph = binomial_random_graph(nproc, p=0.3, seed=1)
W = np.ones((nproc, nproc)) / nproc

# reset local seed
np.random.seed(10*local_rank)

agent = Agent(
    in_neighbors=np.nonzero(graph[local_rank, :])[0].tolist(),
    out_neighbors=np.nonzero(graph[:, local_rank])[0].tolist(),
    in_weights=W[local_rank, :].tolist()
)

# Initialize local function
n = 2
x = Variable(n)
P = np.random.rand(n, n)
P = P.transpose() @ P
bias = np.random.rand(n, 1)
fn = QuadraticForm(x - bias, P)

# Save function on file
with open(f"agent_{agent.id}_function.pkl", "wb") as output:
    pickle.dump(fn, output, pickle.HIGHEST_PROTOCOL)

pb = Problem(fn)
agent.set_problem(pb)

initial_condition = 10*np.random.rand(n, 1)

algorithm = GradientTracking(
    agent=agent,
    initial_condition=initial_condition,
    enable_log=True
)

# run the algorithm
start_time = time.time()
sequence = algorithm.run(
    iterations=parameters.GT_iterations,
    stepsize=parameters.GT_stepsize,
    verbose=True
)
end_time = time.time()
print(f"Agent {agent.id}: {algorithm.get_result().flatten()}")

np.save("agents.npy", nproc)

agent_sequence_file_name = os.path.join(
    RESULTS_DIR, f"agent_{agent.id}_sequence.npy")
np.save(agent_sequence_file_name, sequence)

print(f"Execution time: {round(end_time - start_time, 4)}s")
