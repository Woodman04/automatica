import numpy as np 
from src.system.dynamic_system import DiscreteTimeSystem
from src.filters.kalman_filter import KalmanFilter
from src.utils.plotter import ObserverPlotter
from src.simulator import Simulator
from src.utils.discretize import Discretize
from pathlib import Path

system_name = "Pendolo4"
statesName = ["Theta", "Yaw"]

current_dir = Path(__file__).resolve().parent
graph_path = current_dir / '..' / 'results' / 'kalman'

a11 = 0
a12 = 1
a21 = -9.81  
a22 = 0

A = np.array([[a11, a12],
              [a21, a22]])

C = np.array([[1, 0]])

delta_t = 0.02
running_time = 15

theta_start = np.deg2rad(45)
yaw_start = 0

start_condition = np.array([theta_start, yaw_start])

theta_believed = np.deg2rad(30)
yaw_believed = 0.1
believed_start = np.array([theta_believed, yaw_believed])

Q = np.array([[ 0.02076974, -0.01752707],
              [-0.01752707,  0.03534471]])

R = np.array([[5.70600972]])

A, B = Discretize.euler(delta_t, A)

pendolo = DiscreteTimeSystem(start_condition, delta_t, A, C, Q_real=Q, R_real=R)

Q_estimated = np.array([[0.02, -0.00],
                        [-0.00, 0.020]])

R_estimated = np.array([[5]])

P_estimated = np.array([[ 1500,  0.00],
                        [ 0.00, 1500]])

predictor = KalmanFilter(pendolo, believed_start, Q_estimated, R_estimated, P_estimated)
simulatore = Simulator(pendolo, predictor)
result = simulatore.run(running_time)
print(result[1], result[2], result[0])
ObserverPlotter.plot_kalman(result[1], result[2], result[0], system_name, statesName, graph_path)


