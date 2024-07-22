import dill as pickle
import numpy as np
import os
import time

from mpi4py import MPI
from disropt.agents import Agent
from disropt.functions import Variable, QuadraticForm
from disropt.problems import Problem
from disropt.utils.graph_constructor import binomial_random_graph

import parameters
from giant_admm import GIANTADMM

RESULTS_DIR = "giant_admm_results"

# Get MPI info
comm = MPI.COMM_WORLD
nproc = comm.Get_size()
local_rank = comm.Get_rank()

# Save number of agents on a file
np.save("agents.npy", nproc)

# Generate random graph
graph: np.ndarray = binomial_random_graph(nproc, p=0.3, seed=1)
np.random.seed(10*local_rank)

# Initialize Agent
agent = Agent(in_neighbors=np.nonzero(graph[local_rank, :])[0].tolist(),
              out_neighbors=np.nonzero(graph[:, local_rank])[0].tolist())

# Initialize local function
n = 2
x = Variable(n)
P = np.random.rand(n, n)
P = P.transpose() @ P
bias = np.random.rand(n, 1)
fn = QuadraticForm(x - bias, P)

with open(f"agent_{agent.id}_function.pkl", "wb") as output:
    pickle.dump(fn, output, pickle.HIGHEST_PROTOCOL)

pb = Problem(fn)
agent.set_problem(pb)

initial_condition = 10*np.random.rand(n, 1)
initial_z_values = {i: 10*np.random.rand(2*n, 1) for i in agent.in_neighbors}

algorithm = GIANTADMM(
    agent,
    initial_condition,
    initial_z_values,
    parameters.GIANT_gamma,
    parameters.GIANT_rho,
    parameters.GIANT_alpha,
    enable_log=True
)

start_time = time.time()
sequence = algorithm.run(
    iterations=parameters.GIANT_iterations,
    stepsize=parameters.GIANT_stepsize,
    verbose=True
)
end_time = time.time()

print(f"Agent {agent.id}: {algorithm.get_result().flatten()}")
agent_sequence_file_name = os.path.join(
    RESULTS_DIR, f"agent_{agent.id}_sequence.npy")
np.save(agent_sequence_file_name, sequence)

print(f"Execution time: {round(end_time - start_time, 4)}s")
