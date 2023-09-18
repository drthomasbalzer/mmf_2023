"""
Author: Thomas Balzer
(c) 2023
Material for MMF Stochastic Analysis - Fall 2023
"""
import numpy as np

import plot_utilities as pu
import mmf_2023_1_portfolio_optimisation as p_opt

"""
Portfolio Optimisation Examples - Lecture #2
"""


def value_function(x, r, lambd, mean_r, var, corr, dimension):

    base_v = x * (1 + r)

    sig_m = np.array([[var * corr] * dimension] * dimension)
    #
    for j in range(dimension):
        sig_m[j][j] = var

    mean_v = [mean_r - r] * dimension
    sig_m_inv = np.linalg.inv(sig_m)
    return base_v + 0.5 * np.matmul(np.transpose(np.matmul(sig_m_inv, mean_v)), mean_v) / lambd


if __name__ == '__main__':

    _r = 0.05
    _x = 1.
    _u = 0.15
    _d = -0.05

    _p = 0.6
    _mean_r = p_opt.expected_value_return(_u, _d, _p)
    _var_r = p_opt.variance_return(_u, _d, _p)

    print('Expected Return: ' + str(_mean_r))
    print('Variance of Return: ' + str(_var_r))

    riskaversion_l = .5
    dimensions = [1, 2, 5, 10, 15]
    correlations = [-0.05 + 0.01 * (k+1) for k in range(90)]

    vfs = [[value_function(_x, _r, riskaversion_l, _mean_r, _var_r, _rho, d) for _rho in correlations]
           for d in dimensions]

    mp = pu.PlotUtilities('Value Function as Function of Correlation', 'correlation', 'None')
    mp.multi_plot(correlations, vfs)

