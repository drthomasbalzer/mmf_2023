"""
Author: Thomas Balzer
(c) 2023
Material for MMF Stochastic Analysis - Fall 2023
"""

import numpy as np

import core_math_utilities as dist
import plot_utilities as pu


def plot_maximising_goal_probability(_time, _timestep, _initial_capital, _target, _b, _r, _sigma):

    size = int(_time / _timestep) - 1
    sample = np.random.normal(0, np.sqrt(_timestep), size)

    path_underlying = []
    path_wealth = []
    path_portfolio = []

    x = [_timestep * k for k in range(size)]

    _theta = (_b - _r) / _sigma
    _y0 = np.sqrt(_time) * dist.standard_normal_inverse_cdf(_initial_capital * np.exp(_r * _time) / _target)

    bm = 0
    # initial values
    path_underlying.append(1.)
    path_wealth.append(_initial_capital)
    _y = path_wealth[0] * np.exp(_r * _time) / _target
    _y_inv = dist.standard_normal_inverse_cdf(_y)
    path_portfolio.append(dist.standard_normal_pdf(_y_inv) / (_y * _sigma * np.sqrt(_time)))
    for j in range(1, size):
        _t_remain = _time - x[j]
        _t_sq_remain = np.sqrt(_t_remain)
        path_underlying.append(path_underlying[j-1] * (1. + _b * _timestep + _sigma * sample[j]))
        bm = bm + sample[j] + _theta * _timestep
        pw = _target * np.exp(- _r * _t_remain) * dist.standard_normal_cdf((bm + _y0) / _t_sq_remain)
        path_wealth.append(pw)
        _y = pw * np.exp(_r * _t_remain) / _target
        _y_inv = dist.standard_normal_inverse_cdf(_y)
        path_portfolio.append(dist.standard_normal_pdf(_y_inv) / (_y * _sigma * _t_sq_remain))

    mp = pu.PlotUtilities("Maximising Probability of Reaching a Goal", 'Time', "None")

    labels = ['Stock Price', 'Wealth Process', 'Portfolio Value']
    mp.sub_plots(x, [path_underlying, path_wealth, path_portfolio], labels, ['red', 'blue', 'green'])


if __name__ == '__main__':

    _initial_capital = 1
    _target_wealth = 1.20

    _time = 2.
    timestep = 0.001

    _b = 0.08
    _r = 0.05
    _sigma = .30

    plot_maximising_goal_probability(_time, timestep, _initial_capital, _target_wealth, _b, _r, _sigma)
