'''
Parameters for fine-tuning the algorithms.
'''

# General
iterations = 2000


# Gradient Tracking
GT_iterations = iterations
def GT_stepsize(k): return 1 / ((k + 1)**0.51)


# GIANT-ADMM
GIANT_iterations = iterations
GIANT_gamma = 0.5
GIANT_rho = 0.1
GIANT_alpha = 0.1
def GIANT_stepsize(k): return 0.05


# ADMM Tracking Gradient
# Actual solution: [0.70619038 0.22220143]
ADMM_iterations = iterations
ADMM_gamma = 0.9
ADMM_rho = 0.3
ADMM_alpha = 0.9
def ADMM_stepsize(k): return 0.5
