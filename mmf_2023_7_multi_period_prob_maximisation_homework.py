"""
Author: Thomas Balzer
(c) 2023
Material for MMF Stochastic Analysis - Fall 2023
"""

import numpy as np
import plot_utilities as pu
import mmf_2023_1_portfolio_optimisation as p_opt
import core_math_utilities as dist

class Functor:

    def value(self, x):

        return 0.

class TargetWealthProb(Functor):

    def __init__(self, target):

        self.target = target

    def value(self, x):

        return 1. if x > target else 0.

class Portfolio:

    def __init__(self, p, u, d, pi, r):

        self.p = p
        self.u = u
        self.d = d
        self.pi = pi
        self.r = r

class PortfolioFunctor:

    def __init__(self, portfolio, base_function):

        self.portfolio = portfolio
        self.base_function = base_function

    def value(self, x):

        pf = self.portfolio
        f = self.base_function
        return (pf.p * f.value(p_opt.portfolio_value(x, pf.pi, pf.u, pf.r))
                + (1 - pf.p) * f.value(p_opt.portfolio_value(x, pf.pi, pf.d, pf.r)))



"""
Multi-Period Maximising Probability to Reach a Goal
"""

# def h_f(x, p, u, d, target):
#
#     return p * np.power(x * (1+u), 2.) + (1-p) * np.power(x * (1+d), 2.)
#     # return p * (1. if x * (1+u) >= target else 0.) + (1-p) * (1. if x * (1+d) >= target else 0.)

if __name__ == '__main__':

    print('Heat Equation in Discrete Time - Homework #7')
    # basic parameters applicable to all questions
    _u = 0.15
    _d = -0.05
    _x = 1.
    _p = 0.5
    target = 1.12
    _r = 0.05

    pf = Portfolio(_p, _u , _d, 1., _r)

    f = TargetWealthProb(target)
    n = 8
    x_values = [0.2 + k * 0.05 for k in range(50)]
    y_values = []
    bf = f
    for m in range(1, n+1):
        bf = PortfolioFunctor(pf, bf)
    y_values.append([bf.value(x) for x in x_values])

    mp = pu.PlotUtilities('Target Probability', 'x', 'y')
    mp.multi_plot(x_values, y_values)
    # number_time_steps = 500
    # sample_size = 10000
    # # time_scalar = 1./np.sqrt(number_time_steps)
    # time_scalar = 1./number_time_steps
    #
    # asset_sample = []
    # for k in range(sample_size):
    #     us = np.random.uniform(0., 1., number_time_steps)
    #     bin_s = [1 if un_s < _p else 0 for un_s in us]
    #     s = 1.
    #     for b in bin_s:
    #         s = s * np.power(1 + _u * time_scalar, b) * np.power(1 + _d * time_scalar, 1-b)
    #     asset_sample.append(s)
    #
    # # mp = pu.PlotUtilities('Asset Distribution', 'Outcome', 'Rel. Occurrence')
    # # mp.plot_multi_histogram([asset_sample], 25, ['red'])
    #
    # x_values = [1. + 0.001 * (k+1) for k in range(200)]
    # y_values = []
    # for x in x_values:
    #     y = sum([1 if x * s > target else 0 for s in asset_sample]) / sample_size
    #     y_values.append(y)
    #
    # mp = pu.PlotUtilities('Target Probability', 'x', 'y')
    # mp.multi_plot(x_values, [y_values])
    #
    # #
    # #
    # #
    # # number_steps = 20
    # # uniform_sample = np.random.uniform(0., 1., number_steps)
    # # return_sample = [(_u if u_r < _p else _d) for u_r in uniform_sample]
    # #
    # # y_ax = [_x]
    # # for k in range(number_steps):
    # #     rs = return_sample[k]
    # #     next_val = y_ax[-1] * (1. + rs)
    # #     y_ax.append(next_val)
    # #
    # # x_label = 'Time'
    # # y_label = 'Path Value'
    # # chart_title = 'Geometric Random Walk Over Time'
    # #
    # # x_ax = [k for k in range(number_steps + 1)]
    # # mp = pu.PlotUtilities(chart_title, x_label, y_label)
    # # mp.multi_plot(x_ax, [y_ax])
    # #
    # #
    # #
    # # '''
    # # function f(x) = 1_{x \geq x_0}
    # # '''
    # # _x0 = 1.25
    # #
    # # x_label = 'x'
    # # y_label = 'Hf(x)'
    # # chart_title = 'Hf(x)'
    # #
    # # x_axis = [(k+1) * 0.01 for k in range(200)]
    # # eps = 1.e-06
    # # y_value = [(h_f(x, _p, _u, _d, _x0) - np.power(x, 2.))/eps for x in x_axis]
    # # mp = pu.PlotUtilities(chart_title, x_label, y_label)
    # # mp.multi_plot(x_axis, [y_value])
    # #
    # #
