import numpy as np

from disropt.agents import Agent
from disropt.algorithms import Algorithm
from typing import Dict, override

from admm_tracking_gradient import ADMMTrackingGradient


class GIANTADMM(ADMMTrackingGradient):
    """GIANT-ADMM

    Args:
        agent (Agent): agent to execute the algorithm
        initial_condition (numpy.ndarray): initial condition
        initial_z (Dict[int, numpy.ndarray]): initial value of the z
            vector, as described in the paper, one for each of the
            agent's neighbors
        gamma (float): gamma parameter as desribed in the paper
        rho (float): rho parameter as desribed in the paper
        alpha (float): alpha parameter as desribed in the paper
        enable_log (bool): True for enabling log

    Attributes:
        agent (Agent): agent to execute the algorithm
        x0 (numpy.ndarray): initial condition
        x (numpy.ndarray): current value of the local solution
        gamma (float): gamma parameter as desribed in the paper
        rho (float): rho parameter as desribed in the paper
        alpha (float): alpha parameter as desribed in the paper
        d (int): number of neighbors of the agent
        shape(tuple): shape of the variable
        enable_log (bool): True for enabling log
        z (Dict[int, numpy.ndarray]): current value of the z auxiliary
            variable
        subgradient (float): subgradient of the objective function,
            calculated at the current local solution
    """

    def inverse_hessian(self, x):
        """Inverse hessian matrix of the objective function at point x

        Args:
            x: point on which to calculate inverse hessian
        """
        return np.linalg.inv(self.agent.problem.objective_function.hessian(x))
