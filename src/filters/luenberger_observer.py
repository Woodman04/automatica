from .state_estimator import StateEstimator
from src.system.dynamic_system import DiscreteTimeSystem
import numpy as np

class LuenbergerObserver(StateEstimator):
    def __init__(self, system : DiscreteTimeSystem, poles : np.array, start_estimated : np.array):
        super().__init__(system, start_estimated)
        self.observability_matrix = system.observability_matrix
        self.check_poles(poles)
        self.L = self.create_matrix_L(poles)

    def create_matrix_L(self, poles : np.array):
        e_n = np.zeros((self.n, 1))
        e_n[-1, 0] = 1
        coeffs = np.poly(poles)
        P_A = np.linalg.matrix_power(self.A, self.n)
        for i in range(1, self.n + 1):
            P_A += coeffs[i] * np.linalg.matrix_power(self.A, self.n - i)
        return P_A @ np.linalg.inv(self.observability_matrix) @ e_n
    

    def correct(self, u_next : np.array, y_next):
        D_u_next = self.D @ u_next if self.D is not None else np.zeros(self.p)
        corrective_term = self.L @ ( y_next - self.C @ self._state_estimate - D_u_next)
        self._state_estimate += corrective_term

    def check_poles(self, poles : np.array):
        if max(poles) > 1 or min(poles) < -1:
            raise ValueError("Un polo è maggore di 1")
        if poles.shape[0] != self.n:
            raise ValueError("Numero di poli scorretto")