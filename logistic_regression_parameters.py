RESULTS_DIR = "logistic_regression_results"

# General
C = 10  # Regularization parameter
iterations = 1000  # Number of iterations

# Gradient-Tracking parameters


def GT_stepsize(k): return 0.01


# ADMM-Tracking Gradient parameters
ADMM_gamma = 0.9
ADMM_rho = 0.3
ADMM_alpha = 0.9
def ADMM_stepsize(k): return 0.1


# GIANT-ADMM parameters
GIANT_gamma = 0.8
GIANT_rho = 0.05
GIANT_alpha = 0.7
def GIANT_stepsize(k): return 0.0055
