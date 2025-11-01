from .state_estimator import StateEstimator
from src.system.dynamic_system import DiscreteTimeSystem
import numpy as np

class KalmanFilter(StateEstimator):
    def __init__(self, system : DiscreteTimeSystem, start_estimated : np.array, q_estimated : np.array, r_estimated : np.array, p_estimated_start : np.array):
        super().__init__(system, start_estimated)
        self.p_corrected = p_estimated_start
        self.p_predicted = None
        self.q_estimated = q_estimated
        self.r_estimated = r_estimated
        self.L = None

    def correct(self, u_next, y_next):
        self.update_p_predicted()
        self.update_L()
        self.update_p_corrected()
        corrective_term = self.L @ (y_next - self.C @ self._state_estimate)
        self._state_estimate += corrective_term

    def update_p_predicted(self):
        self.p_predicted = self.A @ self.p_corrected @ (self.A).T + self.q_estimated

    def update_L(self):
        self.L = self.p_predicted @ (self.C).T @ np.linalg.inv(self.C @ self.p_predicted @ (self.C).T + self.r_estimated)

    def update_p_corrected(self):
        self.p_corrected = self.p_predicted - self.L @ self.C @ self.p_predicted