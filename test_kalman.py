import numpy as np 
from dynamic_system import DiscreteTimeSystem
from kalman import Kalman
from plotter import ObserverPlotter

systemName = "Pendolo3"
statesName = ["Theta", "Yaw"]
graph_path = "Kalman"

a11 = 0
a12 = 1
a21 = -9.81  
a22 = 0

A = np.array([[a11, a12],
              [a21, a22]])

C = np.array([[1, 0]])

delta_t = 0.02

theta_start = np.deg2rad(45)
yaw_start = 0

start_condition = np.array([theta_start, yaw_start])
pendolo = DiscreteTimeSystem(start_condition, delta_t, A, C)

theta_believed = np.deg2rad(30)
yaw_believed = 0.1
believed_start = np.array([theta_believed, yaw_believed])

Q = np.array([[ 0.02076974, -0.01752707],
              [-0.01752707,  0.03534471]])

R = np.array([[5.70600972]])

Q_estimated = np.array([[0.01, 0.00],
                        [0.00, 0.05]])

R_estimated = np.array([[5]])

P_estimated = np.array([[ 10,  0.00],
                        [ 0.00, 10]])

predictor = Kalman(pendolo, believed_start, Q_estimated, R_estimated, P_estimated)
stati, stati_stimati, time = predictor.run(5, Q, R)
ObserverPlotter.plotKalman(stati, stati_stimati, time, systemName, statesName, graph_path)


