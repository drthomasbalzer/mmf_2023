"""
Author: Thomas Balzer
(c) 2023
Material for MMF Stochastic Analysis - Fall 2023
"""

import plot_utilities as pu

"""
Portfolio Optimisation Examples - Lecture #1
"""


def expected_value_return(u, d, p):

    return p * u + (1-p) * d


def variance_return(u, d, p):

    return p * pow(u, 2) + (1-p) * pow(d, 2) - pow(expected_value_return(u, d, p), 2)


def portfolio_value(x, pi, risky_return, r):

    return x * ((1 + r) + pi * (risky_return - r))


def exp_portfolio_value(x, pi, u, d, p, r):

    return portfolio_value(x, pi, expected_value_return(u, d, p), r)


def variance_portfolio_value(x, pi, u, d, p):

    return pi * pi * x * x * variance_return(u, d, p)


def value_function(x, r, lambd, u, d, p):

    er = expected_value_return(u, d, p)
    vr = variance_return(u, d, p)
    return x * (1+r) + 0.5 / lambd * (pow(er - r, 2) / vr)


if __name__ == '__main__':

    _r = 0.05
    _x = 1.
    _u = 0.15
    _d = -0.05

    _p = 0.6
    mean_r = expected_value_return(_u, _d, _p)
    var_r = variance_return(_u, _d, _p)
    print('Expected Return: ' + str(mean_r))
    print('Variance of Return: ' + str(var_r))

    pis = [-5 + 0.1 * k for k in range(200)]
    portfolio_values = [exp_portfolio_value(_x, pi, _u, _d, _p, _r) for pi in pis]

    mp = pu.PlotUtilities('Mean of Portfolio Value as Function of $\pi$', '$\pi$', 'None')
    labels = ['Portfolio Mean']
    colors = ['blue']
    mp.sub_plots(pis, [portfolio_values], labels, colors)

    mp = pu.PlotUtilities('Variance of Portfolio Value as Function of $\pi$', '$\pi$', 'None')
    portfolio_values_var = [variance_portfolio_value(_x, pi, _u, _d, _p) for pi in pis]
    labels = ['Portfolio Variance']
    colors = ['blue']
    mp.sub_plots(pis, [portfolio_values_var], labels, colors)

    mp = pu.PlotUtilities('Mean Variance of Portfolio Value as Function of $\pi$', '$\pi$', 'None')
    labels = ['Mean', 'Variance']
    colors = ['blue', 'red']
    mp.sub_plots(pis, [portfolio_values, portfolio_values_var], labels, colors)

    _lambda = 4
    mp = pu.PlotUtilities('Mean-Variance Portfolio Objective Function of $\pi$', '$\pi$', 'None')
    labels = ['Objective Function']
    colors = ['red']
    mp.sub_plots(pis, [[mp - 0.5 * _lambda * vp for (mp, vp) in zip(portfolio_values, portfolio_values_var)]],
                 labels, colors)

    pi_star = 1/(_lambda * _x) * (mean_r - _r) / var_r

    print('Optimal Portfolio: ' + str(pi_star))

    lambda_values = [0.01 * (k+1) for k in range(200)]
    vf = [value_function(_x, _r, lv, _u, _d, _p) for lv in lambda_values]

    mp = pu.PlotUtilities('Value Function as Function of $\lambda$', '$\lambda$', 'None')
    labels = ['Value Function']
    colors = ['red']
    mp.sub_plots(pis, [vf], labels, colors)
