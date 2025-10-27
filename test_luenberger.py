import numpy as np 
from dynamic_system import DiscreteTimeSystem
from observer import Observer
from plotter import ObserverPlotter

systemName = "Pendolo"
statesName = ["Theta", "Yaw"]
graph_path = "Luenberger"

a11 = 0
a12 = 1
a21 = -9.81  
a22 = 0

A = np.array([[a11, a12],
              [a21, a22]])

C = np.array([[1, 0]]) #ho solo posizione in uscita

delta_t = 0.02

theta_start = np.deg2rad(45)
yaw_start = 0

start_condition = np.array([theta_start, yaw_start])
pendolo = DiscreteTimeSystem(start_condition, delta_t, A, C)

theta_believed = np.deg2rad(30)
yaw_believed = 0.1

pole_1 = 0.9
pole_2 = 0.9
poles = np.array([pole_1, pole_2])

believed_start = np.array([theta_believed, yaw_believed])
observerPendolo = Observer(pendolo, poles, believed_start)

runningTime = 5
stati, stati_stimati, time = observerPendolo.run(runningTime)

ObserverPlotter.plotLuenberger(stati, stati_stimati, time, systemName, statesName, poles, graph_path)




