"""
Author: Thomas Balzer
(c) 2023
Material for MMF Stochastic Analysis - Fall 2023
"""


import numpy as np
import plot_utilities as pu


class BMFunctor:

    def __init__(self, vol):
        self.vol = vol

    def next_sample(self, prev_val, sample):
        return 0.

    def initial_value(self):
        return 0.

    def type(self):
        return 'None'


class BrownianMotion(BMFunctor):

    def next_sample(self, prev_val, sample):
        return prev_val + sample * self.vol

    def initial_value(self):
        return 0.

    def type(self):
        return 'Brownian Motion'


class GeometricBrownianMotion(BMFunctor):

    def next_sample(self, prev_val, sample):
        return prev_val * np.exp(self.vol * sample - 0.5 * self.vol * self.vol)

    def initial_value(self):
        return 1.

    def type(self):
        return 'Geometric Brownian Motion'


def geometric_brownian_motion(_time, _timestep, _number_paths, brownian_motion):

    size = int(_time / _timestep)
    total_sz = size * _number_paths

    sample = np.random.normal(0, 1, total_sz)
    paths = []

    # set up x-axis
    x = [_timestep * k for k in range(size + 1)]

    # plot the trajectory of the process
    i = 0
    for k in range(_number_paths):
        path = [brownian_motion.initial_value()] * (size + 1)
        for j in range(size + 1):
            if j == 0:
                continue  # nothing
            else:
                path[j] = brownian_motion.next_sample(path[j-1], sample[i])
                i = i + 1

        paths.append(path)

    max_paths = [[max(path[0:j]) for j in range(1, len(path)+1)] for path in paths]

    mp = pu.PlotUtilities(r'Paths of ' + str(bM.type()), 'Time', 'Random Walk Value')

    plot_max = True
    if plot_max:
        plot_all_max = False
        if plot_all_max:
            mp.multi_plot(x, max_paths)
        else:  # only the first path and its running maximum
            mp.multi_plot(x, [paths[0], max_paths[0]])
    else:
        mp.multi_plot(x, paths)


if __name__ == '__main__':

    time = 5
    timestep = 0.001
    paths = 1
    volatility = 0.2
    bM = BrownianMotion(volatility * np.sqrt(timestep))
    # bM = GeometricBrownianMotion(volatility * np.sqrt(timestep))
    geometric_brownian_motion(time, timestep, paths, bM)
