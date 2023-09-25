"""
Author: Thomas Balzer
(c) 2023
Material for MMF Stochastic Analysis - Fall 2023
"""

import plot_utilities as pu
import core_math_utilities as dist
import mmf_2023_1_portfolio_optimisation as p_opt
import numpy as np

"""
Portfolio Optimisation Examples - Homework #2
"""


def marginal_utility_dpi(x, pi, risky_return, r):

    return (risky_return - r) / p_opt.portfolio_value(x, pi, risky_return, r)


def exp_marginal_utility(x, pi, u, d, p, r):

    return p * marginal_utility_dpi(x, pi, u, r) + (1-p) * marginal_utility_dpi(x, pi, d, r)


def portfolio_log_utility(x, pi, risky_return, r):

    return np.log(p_opt.portfolio_value(x, pi, risky_return, r))


def exp_utility(x, pi, u, d, p, r):

    return p * portfolio_log_utility(x, pi, u, r) + (1-p) * portfolio_log_utility(x, pi, d, r)


if __name__ == '__main__':

    # basic parameters applicable to all questions
    _r = 0.05
    _x = 1.
    _u = 0.15
    _d = -0.05

    # determine minimum and maximum portfolio proportion to ensure V > 0
    min_portfolio = - (1. + _r) / (_u - _r)
    max_portfolio = - (1. + _r) / (_d - _r)
    print('Minimum Portfolio - ' + str(min_portfolio))
    print('Maximum Portfolio - ' + str(max_portfolio))

    step = 0.1
    n_steps = int((max_portfolio - min_portfolio)/step)
    # pis = [min_portfolio + (k+1) * step for k in range(n_steps-1)]
    pis = [-2. + k * step for k in range(100)]

    _p = 0.6

    print('Utility of Risk-Free Portfolio ' + str(exp_utility(_x, 0., _u, _d, _p, _r)))

    """
    Plotting the expected utility and its derivative as a function of $\pi$
    """
    exp_util = [exp_utility(_x, pi, _u, _d, _p, _r) for pi in pis]
    exp_marg_util = [exp_marginal_utility(_x, pi, _u, _d, _p, _r) for pi in pis]
    colors = ['red', 'blue']
    mp = pu.PlotUtilities('Expected Utility and Derivative as Function of $\pi$', '$\pi$', 'None')
    mp.sub_plots(pis, [exp_util, exp_marg_util], ['Utility', 'Derivative'], colors)

    # Solving numerically where the derivative is zero
    eps = 1e-08
    max_exp_utility_portfolio = pis[np.argmax(exp_util)]
    print ('Maximum Portfolio (Numerically) - ' + str(max_exp_utility_portfolio))

    pi_star = (1+_r)*(_p * (_u-_d) + (_d-_r)) / (_r-_d) / (_u-_r)
    print('Log-Optimal Portfolio: ' + str(pi_star))

    print('Optimal Expected Utility: ' + str(exp_utility(_x, pi_star, _u, _d, _p, _r)))

    print('Monte Carlo Approximation of Value Function...')

    sample_size = 10000
    uniform_sample = np.random.uniform(0., 1., sample_size)
    ev = 0
    for uni in uniform_sample:
        ev = ev + portfolio_log_utility(_x, pi_star, (_u if uni < _p else _d), _r)

    print('MC Expected Optimal Utility: ' + str(ev / sample_size))

    # plotting utility as a function of portfolio
    util_mc = []
    for pi in pis:
        ev = 0
        for uni in uniform_sample:
            ev = ev + portfolio_log_utility(_x, pi, (_u if uni < _p else _d), _r)
        util_mc.append(ev / sample_size)

    mp = pu.PlotUtilities('Expected Utility Exact and MC', '$\pi$', 'None')
    mp.sub_plots(pis, [exp_util, util_mc, [eu - um for eu, um in zip(exp_util, util_mc)]],
                 ['Exact', 'MC', 'Diff'], ['red', 'green', 'blue'])

    print('Moment Matching Approximation of Optimal Utility...')

    mean_p = p_opt.exp_portfolio_value(_x, pi_star, _u, _d, _p, _r)
    var_p = p_opt.variance_portfolio_value(_x, pi_star, _u, _d, _p)

    print('Portfolio Mean: ' + str(mean_p))
    print('Portfolio Variance: ' + str(var_p))

    b = np.sqrt(np.log(var_p/mean_p/mean_p + 1.))

    normal_sample = [dist.standard_normal_inverse_cdf(1. - uni) for uni in uniform_sample]

    values_app = [mean_p * np.exp(b * ns - 0.5 * b * b) for ns in normal_sample]
    utility_app = [np.log(v) for v in values_app]

    print('MM Utility: ' + str(np.average(utility_app)))

    print('Moment Matched Mean(MC): ' + str(np.average(values_app)))
    values_ex = [p_opt.portfolio_value(_x, pi_star, (_u if uni < _p else _d), _r) for uni in uniform_sample]
    print('Mean (MC): ' + str(np.average(values_ex)))

    # scatter plot of the simulated defaults
    colors_sc = ['blue']

    mp = pu.PlotUtilities('Portfolio Value and Approx', 'x', 'y')
    mp.scatter_plot(values_ex, [values_app], ['None'], colors_sc)

    nv = 0
    util_approx = []
    for pi in pis:
        mean_pf = p_opt.exp_portfolio_value(_x, pi, _u, _d, _p, _r)
        var_pf = p_opt.variance_portfolio_value(_x, pi, _u, _d, _p)
        b = np.sqrt(np.log(var_pf / mean_pf / mean_pf + 1.))
        nv = 0
        for ns in normal_sample:
            v = (mean_pf * np.exp(b * ns - 0.5 * b * b))
            nv = nv + np.log(v)
        util_approx.append(nv/sample_size)

    mp = pu.PlotUtilities('Expected Utility Exact vs Moment Matched', '$\pi$', 'None')
    mp.sub_plots(pis, [exp_util, util_approx, [eu - um for eu, um in zip(exp_util, util_approx)]],
                 ['Exact', 'MM', 'Diff'], ['red', 'green', 'blue'])
