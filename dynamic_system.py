import numpy as np 
from utils import Utils

class DiscreteTimeSystem():
    def __init__(self, start_condition : np.array, delta_t : float, A : np.array, C : np.array, B : np.array = None, D : np.array = None):
        self.check_dynamic_system(A, B, C, D)

        self.n = A.shape[0]
        self.m = B.shape[1] if B is not None else 0
        self.p = C.shape[0] 

        #eulero avanti (potrebbe causare instabilità)
        self.A = np.eye(self.n) + A * delta_t 
        self.B = None
        if B is not None:
            self.B = B * delta_t
        self.C = C
        self.D = D

        #aggiungi check dimensione
        self.start_condition = start_condition
        self.delta_t = delta_t
        
        #matrice di osservabilità
        self.O = None
        self.isObservable = self.createObservabilityM()


    def createObservabilityM(self):
        self.O = self.C
        for i in range(1, self.n):
            self.O = np.vstack([self.O, self.C @ np.linalg.matrix_power(self.A, i)])
        return np.linalg.matrix_rank(self.O) == self.n
        
    def check_dynamic_system(self, A, B, C, D):
        if not Utils.is_square(A):
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




    
