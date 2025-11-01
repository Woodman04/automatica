import numpy as np 
from src.system.dynamic_system import DiscreteTimeSystem
from src.filters.luenberger_observer import LuenbergerObserver
from src.utils.plotter import ObserverPlotter
from src.simulator import Simulator
from src.utils.discretize import Discretize
from pathlib import Path

systemName = "Pendolo"
statesName = ["Theta", "Yaw"]


current_dir = Path(__file__).resolve().parent
graph_path = current_dir / '..' / 'results' / 'luenberger'

a11 = 0
a12 = 1
a21 = -9.81  
a22 = 0

A = np.array([[a11, a12],
              [a21, a22]])


C = np.array([[1, 0]]) #ho solo posizione in uscita

delta_t = 0.02
running_time = 5

theta_start = np.deg2rad(45)
yaw_start = 0

start_condition = np.array([theta_start, yaw_start])

A, B = Discretize.euler(delta_t, A)

pendolo = DiscreteTimeSystem(start_condition, delta_t, A, C)

theta_believed = np.deg2rad(30)
yaw_believed = 0.1

pole_1 = -0.2
pole_2 = -0.1
poles = np.array([pole_1, pole_2])

believed_start = np.array([theta_believed, yaw_believed])
observerPendolo = LuenbergerObserver(pendolo, poles, believed_start)

simulatorePendolo = Simulator(pendolo, observerPendolo)
result = simulatorePendolo.run(running_time)
ObserverPlotter.plot_luenberger(result[1], result[2], result[0], systemName, statesName, poles, graph_path)






