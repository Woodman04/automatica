import numpy as np
from dynamic_system import DiscreteTimeSystem

class Kalman():
    def __init__(self, system : DiscreteTimeSystem, start_estimated : np.array, q_estimated : np.array, r_estimated : np.array, p_estimated_start : np.array):
        self.system = system
        self.start_estimated = start_estimated
        self.q_estimated = q_estimated
        self.r_estimated = r_estimated
        self.p_estimated_start = p_estimated_start

    def run(self, runningTime : int, q_real : np.array, r_real : np.array):
        steps = int(runningTime / self.system.delta_t) 
        stati = np.zeros((steps, self.system.n))
        uscita = np.zeros((steps, self.system.p))
        stati_stimati = np.zeros((steps, self.system.n))
        stati[0] = self.system.start_condition
        stati_stimati[0] = self.start_estimated
        mean_vx = np.zeros(self.system.n)
        mean_vy = np.zeros(self.system.p)
        vy = np.random.multivariate_normal(mean_vy, r_real, size=1)
        D_u_k = self.system.D @ input[0] if self.system.D else np.zeros(self.system.p)
        uscita[0] = self.system.C @ stati[0] + D_u_k + vy
        p = self.p_estimated_start
        #predictor
        for i in range(1, steps):
            B_u_k = self.system.B @ input[i - 1] if self.system.B else np.zeros(self.system.n)
            vx = np.random.multivariate_normal(mean_vx, q_real, size=1)
            stati[i] = self.system.A @ stati[i - 1] + B_u_k + vx
            L = self.system.A @ p @ (self.system.C).T @ np.linalg.inv(self.system.C @ p @ (self.system.C).T + self.r_estimated)
            p = self.system.A @ p @ (self.system.A).T + self.q_estimated - self.system.A @ p @ (self.system.C).T @ np.linalg.inv(self.system.C @ p @ (self.system.C).T + self.r_estimated) @ self.system.C @ p @ (self.system.A).T
            stati_stimati[i] = self.system.A @ stati_stimati[i - 1] + B_u_k + L @ (uscita[i - 1] - self.system.C @ stati_stimati[i - 1] - D_u_k) 
            D_u_k = self.system.D @ input[i] if self.system.D else np.zeros(self.system.p)
            vy = np.random.multivariate_normal(mean_vy, r_real, size=1)
            uscita[i] = self.system.C @ stati[i] + D_u_k + vy
        time = np.linspace(0, runningTime, steps)
        return stati, stati_stimati, time

        