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
Portfolio Optimisation Examples - Lecture #2
"""


def probability_exceeding_goal(x, pi, u, d, p, r, goal):

    return (p * (1. if p_opt.portfolio_value(x, pi, u, r) >= goal else 0.)
            + (1 - p) * (1. if p_opt.portfolio_value(x, pi, d, r) >= goal else 0.))


def probability_exceeding_goal_mm_normal(x, pi, u, d, p, r, goal):

    mean_p = p_opt.exp_portfolio_value(x, pi, u, d, p, r)
    var_p = p_opt.variance_portfolio_value(x, pi, u, d, p)
    nd = dist.NormalDistribution(mean_p, var_p)
    return 1 - nd.cdf(goal) if var_p > 0 else 0.


def probability_exceeding_goal_mm_ln(x, pi, u, d, p, r, goal):

    mean_p = p_opt.exp_portfolio_value(x, pi, u, d, p, r)
    var_p = p_opt.variance_portfolio_value(x, pi, u, d, p)
    b = np.sqrt(np.log(var_p / mean_p / mean_p + 1.))
    nd = dist.NormalDistribution(0., 1.)
    c = (np.log(goal / mean_p) + 0.5 * b * b)/b
    return 1. - nd.cdf(c)


if __name__ == '__main__':

    _r = 0.05
    _x = 1.
    _u = 0.15
    _d = -0.05

    # Maximising the Probability of Reaching a Goal
    x_0 = 1.08

    min_portfolio = - (1. + _r) / (_u - _r)
    max_portfolio = - (1. + _r) / (_d - _r)
    print('Minimum Portfolio - ' + str(min_portfolio))
    print('Maximum Portfolio - ' + str(max_portfolio))
    step = 0.1
    n_steps = int((max_portfolio - min_portfolio)/step)
    pi_all = [min_portfolio + (k+1) * step for k in range(n_steps-1)]
    pis = [-1 + 0.1 * k * step for k in range(300)]

    _p = 0.6

    p_exact = [probability_exceeding_goal(_x, pi, _u, _d, _p, _r, x_0) for pi in pi_all]
    p_mm_n = [probability_exceeding_goal_mm_normal(_x, pi, _u, _d, _p, _r, x_0) for pi in pi_all]
    p_mm_ln = [probability_exceeding_goal_mm_ln(_x, pi, _u, _d, _p, _r, x_0) for pi in pi_all]

    print('Maximum Probabilities ~~~')
    print('Exact: ' + str(max(p_exact)))
    print('Normal Moment Matching: ' + str(max(p_mm_n)))
    print('Lognormal Moment Matching: ' + str(max(p_mm_ln)))

    mp = pu.PlotUtilities('Probability of Reaching Goal as Function of $\pi$', '$\pi$', 'None')
    mp.multi_plot(pi_all, [p_exact])

    p_exact = [probability_exceeding_goal(_x, pi, _u, _d, _p, _r, x_0) for pi in pis]
    mp = pu.PlotUtilities('Probability of Reaching Goal as Function of $\pi$', '$\pi$', 'None')
    mp.multi_plot(pis, [p_exact])

    p_mm_n = [probability_exceeding_goal_mm_normal(_x, pi, _u, _d, _p, _r, x_0) for pi in pis]
    p_mm_ln = [probability_exceeding_goal_mm_ln(_x, pi, _u, _d, _p, _r, x_0) for pi in pis]
    mp = pu.PlotUtilities('Probability of Reaching Goal as Function of $\pi$ - Exact vs MM', '$\pi$', 'None')
    mp.multi_plot(pis, [p_exact, p_mm_n])

    mp = pu.PlotUtilities('Probability of Reaching Goal as Function of $\pi$ - Exact vs MM(Normal/LN)', '$\pi$', 'None')
    mp.multi_plot(pis, [p_exact, p_mm_n, p_mm_ln])
