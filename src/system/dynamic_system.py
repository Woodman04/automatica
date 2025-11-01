import numpy as np 
from src.utils.matrix import MatrixUtils

class DiscreteTimeSystem():
    def __init__(self, start_condition : np.array, delta_t : float, A : np.array, C : np.array, B : np.array = None, D : np.array = None, Q_real : np.array = None, R_real : np.array = None):
        self.check_dynamic_system(A, B, C, D, start_condition)

        self._state = start_condition
        
        self.n = A.shape[0]
        self.m = B.shape[1] if B is not None else 0
        self.p = C.shape[0] 

        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.Q = Q_real
        self.R = R_real
        self.delta_t = delta_t
        
        #matrice di osservabilità
        self.observability_matrix = None
        self.isObservable = self.create_observability_matrix()

    @property
    def state(self):
        return self._state

    def step(self, u : np.array):
        C_x_k = self.C @ self._state
        D_u_k = self.D @ u if self.D is not None else np.zeros(self.p)
        gaussian_noise_y = np.random.multivariate_normal(np.zeros(self.p), self.R, size=1).flatten() if self.R is not None else np.zeros(self.p)
        y_k = C_x_k + D_u_k + gaussian_noise_y
        A_x_k = self.A @ self._state
        B_u_k = self.B @ u if self.B is not None else np.zeros(self.n)
        gaussian_noise_x = np.random.multivariate_normal(np.zeros(self.n), self.Q, size=1).flatten() if self.Q is not None else np.zeros(self.n)
        self._state = A_x_k + B_u_k + gaussian_noise_x
        return y_k


    def create_observability_matrix(self):
        self.observability_matrix = self.C 
        for i in range(1, self.n):
            self.observability_matrix = np.vstack([self.observability_matrix, self.C @ np.linalg.matrix_power(self.A, i)])
        self.observability_matrix = self.observability_matrix @ self.A
        return np.linalg.matrix_rank(self.observability_matrix) == self.n
        
    def check_dynamic_system(self, A, B, C, D, start_condition):
        if not MatrixUtils.is_square(A):
            raise ValueError(f"A non è quadrata: {A.shape}")
        n = A.shape[0]
        if B is not None:
            if B.shape[0] != n:
                raise ValueError(f"B non compatibile con A: {B.shape}")
        if C.shape[1] != n:
            raise ValueError(f"C non compatibile con A: {C.shape}")
        if D is not None:
            if D.shape[0] != C.shape[0] or D.shape[1] != B.shape[1]:
                raise ValueError(f"D non compatibile con C e B: {D.shape}")
        if start_condition.shape[0] != n:
            raise ValueError("Stato iniziale non compatibile con A")




    
