from src.system.dynamic_system import DiscreteTimeSystem
from src.filters.state_estimator import StateEstimator
import numpy as np

class Simulator():
    def __init__(self, system : DiscreteTimeSystem, estimator : StateEstimator ):
        self.system = system
        self.estimator = estimator
        self.n = self.system.n
        self.m = self.system.m

    def run(self, running_time : float, input_sequence : np.array = None):
        delta_t = self.system.delta_t
        steps = int(running_time / delta_t)

        if input_sequence is None:
            input_sequence = np.zeros((steps, self.m))

        states = np.zeros((steps, self.n))
        estimated_states = np.zeros((steps, self.n))
        states[0] = self.system.state
        estimated_states[0] = self.estimator.state_estimate

        self.system.step(input_sequence[0])
        for i in range(1, steps):
            states[i] = self.system.state
            self.estimator.predict(input_sequence[i - 1])
            y = self.system.step(input_sequence[i])
            self.estimator.correct(input_sequence[i], y)
            estimated_states[i] = self.estimator.state_estimate

        time = np.linspace(0, running_time, steps)
        result = (time, states, estimated_states)
        return result
        



