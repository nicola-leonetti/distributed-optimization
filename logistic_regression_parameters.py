RESULTS_DIR = "logistic_regression_results"

# Gradient-Tracking parameters


def GT_stepsize(k): return 0.001


# ADMM-Tracking Gradient parameters
ADMM_gamma = 0.9
ADMM_rho = 0.3
ADMM_alpha = 0.9
def ADMM_stepsize(k): return 0.05


# GIANT-ADMM parameters
GIANT_gamma = 0.5
GIANT_rho = 0.9
GIANT_alpha = 0.9
def GIANT_stepsize(k): return 0.5
