from abc import ABC, abstractmethod
from src.system.dynamic_system import DiscreteTimeSystem
import numpy as np

class StateEstimator(ABC):
    def __init__(self, system : DiscreteTimeSystem, start_estimated : np.array):
        self.check(system, start_estimated)
        self.A = system.A
        self.B = system.B
        self.C = system.C
        self.D = system.D
        self.n = self.A.shape[0]
        self.m = self.B.shape[1] if self.B is not None else 0
        self.p = self.C.shape[0]
        self._state_estimate = start_estimated

    def check(self, system : DiscreteTimeSystem, start_estimated : np.array):
        if not system.isObservable:
            raise ValueError("Il sistema non è osservabile")
        if start_estimated.shape[0] != system.n:
            raise ValueError("Numero di stati iniziali scorretto")
    
    @property
    def state_estimate(self):
        return self._state_estimate

    def predict(self, u_k : np.array):
        A_x_k = self.A @ self._state_estimate
        B_u_k = self.B @ u_k if self.B is not None else np.zeros(self.n)
        self._state_estimate = A_x_k + B_u_k

    @abstractmethod
    def correct(self, u_next, y_next):
        pass