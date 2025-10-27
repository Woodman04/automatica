from dynamic_system import DiscreteTimeSystem
import numpy as np

class Observer():
    def __init__(self, system : DiscreteTimeSystem, poles : np.array, supposedStart : np.array):
        self.check(system, poles, supposedStart)
        self.system = system
        #aggiungi check
        self.supposedStart = supposedStart
        self.L = self.createMatrixL(poles)

    def createMatrixL(self, poles : np.array):
        e_n = np.zeros((self.system.n, 1))
        e_n[-1, 0] = 1
        coeffs = np.poly(poles)
        P_A = np.linalg.matrix_power(self.system.A, self.system.n)
        for i in range(1, self.system.n + 1):
            P_A += coeffs[i] * np.linalg.matrix_power(self.system.A, self.system.n - i)
        return P_A @ np.linalg.inv(self.system.O) @ e_n
    
    def run(self, runningTime : int, input : np.array = None):
        steps = int(runningTime / self.system.delta_t) 
        stati = np.zeros((steps, self.system.n))
        uscita = np.zeros((steps, self.system.p))
        stati_stimati = np.zeros((steps, self.system.n))
        stati[0] = self.system.start_condition
        stati_stimati[0] = self.supposedStart
        D_u_k = self.system.D @ input[0] if self.system.D else np.zeros(self.system.p)
        uscita[0] = self.system.C @ stati[0] + D_u_k
        #predictor
        for i in range(1, steps):
            B_u_k = self.system.B @ input[i - 1] if self.system.B else np.zeros(self.system.n)
            stati[i] = self.system.A @ stati[i - 1] + B_u_k
            stati_stimati[i] = self.system.A @ stati_stimati[i - 1] + B_u_k + self.L @ (uscita[i - 1] - self.system.C @ stati_stimati[i - 1] - D_u_k) 
            D_u_k = self.system.D @ input[i] if self.system.D else np.zeros(self.system.p)
            uscita[i] = self.system.C @ stati[i] + D_u_k
        time = np.linspace(0, runningTime, steps)
        return stati, stati_stimati, time



    def check(self, system, poles, supposedStart):
        if not system.isObservable:
            raise ValueError("Il sistema non è osservabile")
        if max(poles) > 1 or min(poles) < -1:
            raise ValueError("Un polo è maggore di 1")
        if poles.shape[0] != system.n:
            raise ValueError("Numero di poli scorretto")
        
    




