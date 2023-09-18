"""
Author: Thomas Balzer
(c) 2023
Material for MMF Stochastic Analysis - Fall 2023
"""

import plot_utilities as pu
import mmf_2023_1_portfolio_optimisation as p_opt

"""
Portfolio Optimisation Examples - Homework #1
"""


if __name__ == '__main__':

    # basic parameters applicable to all questions
    _r = 0.05
    _x = 1.
    _u = 0.15
    _d = -0.05

    probs = [0.01 * (k+1) for k in range(98)]
    exp_values = [p_opt.expected_value_return(_u, _d, pr) for pr in probs]
    variances = [p_opt.variance_return(_u, _d, pr) for pr in probs]

    mp = pu.PlotUtilities('Mean and Variance of Return as Function of $p$', '$p$', 'None')
    labels = ['Mean', 'Variance']
    colors = ['red', 'blue']
    mp.sub_plots(probs, [exp_values, variances], labels, colors)

    pis = [0.1 * k for k in range(200)]
    _p = 0.6
    portfolio_values = [p_opt.exp_portfolio_value(_x, pi, _u, _d, _p, _r) for pi in pis]
    variance_portfolio = [p_opt.variance_portfolio_value(_x, pi, _u, _d, _p) for pi in pis]

    mp = pu.PlotUtilities('Mean and Variance of Portfolio Value as Function of $\pi$', '$\pi$', 'None')
    mp.sub_plots(pis, [portfolio_values, variance_portfolio], labels, colors)

    mean_var_optimal_portfolio = ((1 / 1. / _x) *
                                  (p_opt.expected_value_return(_u, _d, _p) - _r) / p_opt.variance_return(_u, _d, _p))
    print('Mean-Variance Optimal Portfolio ' + str(mean_var_optimal_portfolio))

    mp = pu.PlotUtilities('Value Function', '$p$', 'None')
    mp.multi_plot(probs, [[p_opt.value_function(_x, _r, 1., _u, _d, pr) for pr in probs]])
