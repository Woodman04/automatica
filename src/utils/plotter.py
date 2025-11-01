import numpy as np
import matplotlib.pyplot as plt
import os

class ObserverPlotter():
    @staticmethod
    def plot_luenberger(states : np.array , estimated_states : np.array, time : np.array, graph_name : str, states_name : list[str], poles = None, save_path : str = None):
        fig, ax = plt.subplots(states.shape[1], 1, figsize=(10, 8))
        poles_str = ", ".join([f"{p:.2f}" for p in poles])
        fig.suptitle(f"Stima dell\'Osservatore Luenberger ({graph_name}) con poli {poles_str}", fontsize=16)
        for i in range(states.shape[1]):
            ax[i].plot(time, states[:, i], 'b-', label=f"{states_name[i]} Reale ($\theta$)")
            ax[i].plot(time, estimated_states[:, i], 'r--', label=f"{states_name[i]} Stimato ($\hat{{\theta}}$)")
            ax[i].set_ylabel('Angolo (rad)')
            ax[i].legend()
            ax[i].grid(True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        ObserverPlotter.save_luenberger_graph(save_path, graph_name, poles_str)
        plt.show()

    @staticmethod
    def plot_kalman(states : np.array , estimated_states : np.array, time : np.array, graph_name : str, states_name : list[str], save_path : str = None):
        fig, ax = plt.subplots(states.shape[1], 1, figsize=(10, 8))
        fig.suptitle(f"Kalman({graph_name})", fontsize=16)
        for i in range(states.shape[1]):
            ax[i].plot(time, states[:, i], 'b-', label=f"{states_name[i]} Reale ($\theta$)")
            ax[i].plot(time, estimated_states[:, i], 'r--', label=f"{states_name[i]} Stimato ($\hat{{\theta}}$)")
            ax[i].set_ylabel('Angolo (rad)')
            ax[i].legend()
            ax[i].grid(True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        ObserverPlotter.save_kalman_graph(save_path, graph_name)
        plt.show()
    
    @staticmethod
    def save_luenberger_graph(save_path, graph_name, poles):
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            filename = f"Luenberger_{graph_name}_{poles}.png"
            full_path = os.path.join(save_path, filename)
            plt.savefig(full_path, bbox_inches='tight', dpi=300)
            print(f"Grafico salvato in: {full_path}")

    @staticmethod        
    def save_kalman_graph(save_path, graph_name):
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            filename = f"Kalman_{graph_name}.png"
            full_path = os.path.join(save_path, filename)
            plt.savefig(full_path, bbox_inches='tight', dpi=300)
            print(f"Grafico salvato in: {full_path}")



            


