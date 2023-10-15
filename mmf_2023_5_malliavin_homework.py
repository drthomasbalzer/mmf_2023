"""
Author: Thomas Balzer
(c) 2023
Material for MMF Stochastic Analysis - Fall 2023
"""

import numpy as np
import plot_utilities as pu
import mmf_2023_2_portfolio_optimisation_homework as p_opt
import core_math_utilities as dist


"""
Malliavin Calculus - Homework #5
"""


def realised_return(a, b, x):

    return a * np.exp(b * x - 0.5 * b * b) - 1.


def _dens_argument(a, b, x):

    return np.log((1.+x)/a)/b + 0.5 * b


def return_density(a, b, x):

    return dist.standard_normal_pdf(_dens_argument(a, b, x)) / (b * (1 + x))


def return_density_log_deriv(a, b, x):

    return _dens_argument(a, b, x) / (a * b)


if __name__ == '__main__':

    # basic parameters applicable to all questions
    _r = 0.05
    _x = 1.
    _b = 0.08
    _pi = 2

    pert = 1.e-08

    rf_r = 1. + _r
    a_range = [1. + k * 0.001 for k in range(250)]
    n_repeats = 5000
    normal_sample = np.random.normal(0., 1., n_repeats)

    '''
    Sample of returns for a fixed a
    '''
    return_sample = [realised_return(1.1, _b, ns) for ns in normal_sample]
    mp = pu.PlotUtilities("Histogram of Returns for $a = 1.1$", 'Outcome', 'Rel. Occurrence')
    mp.plot_histogram(return_sample, 40, ['Return'])

    '''
    Visualisation of return density and its derivative (again for fixed a)
    '''
    _a = 1.1
    r_range = [-0.25 + 0.005 * (k+1) for k in range(400)]
    d_values = [return_density(_a, _b, rr) for rr in r_range]
    deriv_values = [(return_density(_a + pert, _b, rr) - return_density(_a, _b, rr)) / pert for rr in r_range]
    deriv_values_exp = [return_density_log_deriv(1.1, _b, rr) * return_density(_a, _b, rr) for rr in r_range]

    mp = pu.PlotUtilities('Density Derivative - Analytic and Bump & Reval', '$x$', 'None')
    mp.sub_plots(r_range, [deriv_values, deriv_values_exp], ['B&R', 'Explicit'], ['red', 'green'])

    exp_utility = []
    for _a in a_range:
        realised_utility = [p_opt.portfolio_log_utility(_x, _pi, realised_return(_a, _b, ns), _r)
                            for ns in normal_sample]
        exp_util = sum(realised_utility) / n_repeats
        exp_utility.append(exp_util)

    mp = pu.PlotUtilities('Expected Utility as Function of $a$', '$a$', 'None')
    mp.multi_plot(a_range, [exp_utility])

    marg_utility = []
    '''
    Derivative of Expected Utility with Respect to Mean $a$ (Bump and Reval)
    '''
    for _a in a_range:
        realised_utility_pert = [p_opt.portfolio_log_utility(_x, _pi, realised_return(_a + pert, _b, ns), _r)
                                 for ns in normal_sample]
        realised_utility = [p_opt.portfolio_log_utility(_x, _pi, realised_return(_a, _b, ns), _r)
                            for ns in normal_sample]
        margin_exp_util = (sum(realised_utility_pert) - sum(realised_utility)) / n_repeats / pert
        marg_utility.append(margin_exp_util)

    mp = pu.PlotUtilities('Expected Utility and Derivative as Function of $a$', '$a$', 'None')
    mp.sub_plots(a_range, [exp_utility, marg_utility], ['Utility', 'Derivative'], ['red', 'green'])

    exp_utility = []
    exp_utility_deriv = []

    '''
    Derivative of Expected Utility with Respect to Mean $a$ - In Path Derivative
    '''
    for _a in a_range:
        realised_returns = [realised_return(_a, _b, ns) for ns in normal_sample]
        realised_utility = [p_opt.portfolio_log_utility(_x, _pi, _rr, _r) for _rr in realised_returns]
        realised_utility_deriv = [p_opt.portfolio_log_utility(_x, _pi, _rr, _r)
                                  * return_density_log_deriv(_a, _b, _rr)
                                  for _rr in realised_returns]
        exp_utility.append(sum(realised_utility) / n_repeats)
        exp_utility_deriv.append(sum(realised_utility_deriv) / n_repeats)

    mp = pu.PlotUtilities('Expected Utility and Derivative as Function of $a$', '$a$', 'None')
    mp.sub_plots(a_range, [exp_utility, exp_utility_deriv], ['Utility', 'Derivative'], ['red', 'green'])
