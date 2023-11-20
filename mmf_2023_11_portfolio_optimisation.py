"""
Author: Thomas Balzer
(c) 2023
Material for MMF Stochastic Analysis - Fall 2023
"""

import numpy as np
import plot_utilities as pu


def terminal_utility_histogram(_b, _r, _sigma, _t, _sample_size):

    # Plots the terminal distribution of a portfolio
    normal_sample = np.random.normal(0, _t, _sample_size)

    alpha = .0
    pi = (_b - _r) / (_sigma * _sigma) / (1. - alpha)
    sigma_pi = _sigma * pi
    b_pi = _r + pi * (_b - _r)

    sigmas = [_sigma, sigma_pi, _sigma * 0.4]
    returns = [_b, b_pi, 0.4 * _b + 0.6 * _r]
    colors = ['red', 'green', 'blue']
    samples = []

    for s, ret in zip(sigmas[:2], returns[:2]):
        sample = [np.exp((ret - 0.5 * s * s) * _t + s * ns) for ns in normal_sample]
        samples.append(sample)

    # print(np.average(sample_value_stock))
    num_bins = 100

    mp = pu.PlotUtilities("Terminal Wealth for Stock and Mixed Portfolio for $\pi=${0}".format(pi),
                          'Outcome', 'Rel. Occurrence')
    mp.plot_multi_histogram(samples, num_bins, colors)

    # a_star = 0.10
    # alphas = [0.01 * k for k in range(20)]
    # utility = []
    # for _a in alphas:
    #     pi = (_b - _r) / (_sigma * _sigma) / (1. - _a)
    #     sigma_pi = _sigma * pi
    #     b_pi = _r + pi * (_b - _r)
    #     exp_utility = sum([1./a_star * np.exp(a_star *
    #     ((b_pi - 0.5 * sigma_pi * sigma_pi) * _t + sigma_pi * ns)) for ns in normal_sample]) / _sample_size
    #     utility.append(exp_utility)
    #
    # print(np.max(utility))
    # mp = pu.PlotUtilities('Expected Utility for $a^*={0}$'.format(a_star), 'x', 'y')
    # mp.multi_plot(alphas, [utility])


if __name__ == '__main__':

    _b = .2
    _sigma = 0.5
    _r = 0.05
    _t = 1.
    _n = 25000

    terminal_utility_histogram(_b, _r, _sigma, _t, _n)
