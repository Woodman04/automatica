import numpy as np
import matplotlib.pyplot as plt
import os

class ObserverPlotter():
    @staticmethod
    def plotLuenberger(states : np.array , estimated_states : np.array, time : np.array, nomeGrafico : str, statesName : list[str], poles = None, save_path : str = None):
        fig, ax = plt.subplots(states.shape[1], 1, figsize=(10, 8))
        poles_str = ", ".join([f"{p:.2f}" for p in poles])
        fig.suptitle(f"Stima dell\'Osservatore ({nomeGrafico}) con poli {poles_str}", fontsize=16)
        for i in range(states.shape[1]):
            ax[i].plot(time, states[:, i], 'b-', label=f"{statesName[i]} Reale ($\theta$)")
            ax[i].plot(time, estimated_states[:, i], 'r--', label=f"{statesName[i]} Stimato ($\hat{{\theta}}$)")
            ax[i].set_ylabel('Angolo (rad)')
            ax[i].legend()
            ax[i].grid(True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        ObserverPlotter.saveL(save_path, nomeGrafico, poles_str)
        plt.show()

    @staticmethod
    def plotKalman(states : np.array , estimated_states : np.array, time : np.array, nomeGrafico : str, statesName : list[str], save_path : str = None):
        fig, ax = plt.subplots(states.shape[1], 1, figsize=(10, 8))
        fig.suptitle(f"Stima di\'Kalman({nomeGrafico})", fontsize=16)
        for i in range(states.shape[1]):
            ax[i].plot(time, states[:, i], 'b-', label=f"{statesName[i]} Reale ($\theta$)")
            ax[i].plot(time, estimated_states[:, i], 'r--', label=f"{statesName[i]} Stimato ($\hat{{\theta}}$)")
            ax[i].set_ylabel('Angolo (rad)')
            ax[i].legend()
            ax[i].grid(True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        ObserverPlotter.saveK(save_path, nomeGrafico)
        plt.show()
    
    @staticmethod
    def saveL(save_path, nomeGrafico, poles):
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            filename = f"Osservatore_{nomeGrafico}_{poles}.png"
            full_path = os.path.join(save_path, filename)
            plt.savefig(full_path, bbox_inches='tight', dpi=300)
            print(f"Grafico salvato in: {full_path}")

    @staticmethod        
    def saveK(save_path, nomeGrafico):
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            filename = f"Kalman_{nomeGrafico}.png"
            full_path = os.path.join(save_path, filename)
            plt.savefig(full_path, bbox_inches='tight', dpi=300)
            print(f"Grafico salvato in: {full_path}")



            


