import numpy as np

from typing import Callable, Dict, Union, override

from disropt.agents.agent import Agent
from disropt.algorithms import Algorithm
from disropt.problems import Problem

import parameters


class ADMMTrackingGradient(Algorithm):
    """ADMM-Tracking Gradient

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

    @override
    def __init__(self,
                 agent: Agent,
                 initial_condition: np.ndarray,
                 initial_z: np.ndarray,
                 gamma: float = 0.001,
                 rho: float = 0.1,
                 alpha: float = 0.1,
                 enable_log: bool = False):

        super(ADMMTrackingGradient, self).__init__(agent, enable_log)

        if not isinstance(initial_condition, np.ndarray):
            raise TypeError("Initial condition must be a Numpy vector")
        self.x0 = initial_condition
        self.x = initial_condition

        if not isinstance(gamma, float):
            raise TypeError("Tuning gain gamma must be a float")
        elif gamma <= 0:
            raise ValueError("Tuning gain gamma must be > 0")
        self.gamma = gamma

        if not isinstance(rho, float):
            raise TypeError("Tuning parameter rho must be a float")
        elif rho <= 0:
            raise ValueError("Tuning parameter rho must be > 0")
        self.rho = rho

        if not isinstance(alpha, float):
            raise TypeError("Tuning parameter alpha must be a float")
        elif alpha <= 0 or alpha >= 1:
            raise ValueError(
                "Tuning parameter alpha must be between 0 and 1 excluded")
        self.alpha = alpha

        if not isinstance(agent.problem, Problem):
            raise TypeError("The agent must be equipped with a Problem")

        # Number of neighbors of i-th agent
        if self.agent.in_neighbors != self.agent.out_neighbors:
            raise ValueError("The graph must be undirected")
        self.d = len(self.agent.in_neighbors)

        self.shape = self.x.shape

        # The dict maps the j-th neighbor to z_ij
        if not isinstance(initial_z, dict):
            raise TypeError("Initial z value must be a dict[int, np.ndarray]")
        self.z: Dict[int, np.ndarray] = initial_z

        self.subgradient = self.agent.problem.objective_function.subgradient

    def _update_local_solution(self, x: np.ndarray, **kwargs):
        """update the local solution x

        Args:
            x: new value

        Raises:
            TypeError: Input must be a numpy.ndarray
            ValueError: Incompatible shapes
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a numpy.ndarray")
        if x.shape != self.x.shape:
            raise ValueError("Incompatible shapes")
        self.x = x

    def iterate_run(self, stepsize: float, **kwargs):
        """Run a single iteration of the algorithm

        Args: 
            stepsize: If a float is given as input, the stepsize is 
                constant. If a function is given, it must take an 
                iteration k as input and output the corresponding 
                stepsize. Defaults to 0.1.
        """
        # Update y and s (online auxiliary optimization problem with
        # ADMM)
        const = 1 / (1 + self.rho*self.d)
        sum_z = np.sum(list(self.z.values()), axis=0)
        half_len = len(sum_z) // 2
        self.y = const * (self.x + sum_z[:half_len])
        self.s = const * (self.subgradient(self.x) + sum_z[half_len:])

        # Control law
        # (In order to avoid code duplication in the GIANTADMM class, we
        # check if a inverse_hessian has been defined in the subclass.
        # If not, we run the algorithm normally).
        if getattr(self, "inverse_hessian", None) is not None:
            u = self.gamma*(self.y - self.x - stepsize *
                            self.inverse_hessian(self.x) @ self.s)
        else:
            u = self.gamma*(self.y - self.x - stepsize * self.s)

        self._update_local_solution(self.x + u, **kwargs)

        # Exchange messages with neighbors
        m: Dict[int, np.ndarray] = {}
        for j, z in self.z.items():
            m[j] = -z + 2*self.rho*np.concatenate((self.y, self.s))
        m = self.agent.neighbors_exchange(m)

        # Update z according to the messages received
        for j, z in self.z.items():
            self.z[j] = (1 - self.alpha)*z + self.alpha*m[j][self.agent.id]

    def run(self,
            iterations: int = 1000,
            stepsize: Union[float, Callable] = 0.1,
            verbose: bool = False
            ) -> np.ndarray:
        """Run the algorithm for a given number of iterations

        Args:
            iterations: Number of iterations. Defaults to 1000.
            stepsize: If a float is given as input, the stepsize is 
                constant. If a function is given, it must take an 
                iteration k as input and output the corresponding 
                stepsize. Defaults to 0.1.
            verbose: If True print some information during the evolution
                of the algorithm. Defaults to False.

        Raises:
            TypeError: The number of iterations must be an int
            TypeError: The stepsize must be a float or a callable

        Returns:
            return a tuple (x, lambda) with the sequence of primal solutions and dual variables if enable_log=True.
        """

        if not isinstance(iterations, int):
            raise TypeError("Number of iterations must be of type int")
        if not (isinstance(stepsize, float) or callable(stepsize)):
            raise TypeError("The stepsize must be a float or a function")

        if self.enable_log:
            self.sequence = np.zeros([iterations] + list(self.shape))

        for k in range(iterations):
            step = stepsize(k) if callable(stepsize) else stepsize
            self.iterate_run(stepsize=step)

            if self.enable_log:
                self.sequence[k] = self.x
            # Only once per iteration and only if in verbose mode
            if verbose and self.agent.id == 0:
                print(f"Iteration {k}", end="\r")

        if self.enable_log:
            return self.sequence

    def get_result(self):
        """Return the actual value of x

        Returns:
            numpy.ndarray: value of x
        """
        return self.x
