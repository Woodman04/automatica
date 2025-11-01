import numpy as np

class Discretize():
    @staticmethod
    def euler( delta_t : float, A : np.array, B : np.array = None):
        A = np.eye(A.shape[0]) + A * delta_t
        B = B * delta_t if B is not None else None
        return A, B