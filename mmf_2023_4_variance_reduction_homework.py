"""
Author: Thomas Balzer
(c) 2023
Material for MMF Stochastic Analysis - Fall 2023
"""

import numpy as np
import core_math_utilities as dist
import plot_utilities as pu

"""
Variance Reduction - Homework #4
"""


if __name__ == '__main__':

    # basic parameters applicable to all questions
    _rf_rate = 0.05
    _x = 1.
    _x0 = 1.1
    a = 1.08
    b = 0.08

    rf_r = 1. + _rf_rate
    # determine the limit for $\pi -> \infty$
    limit_prob = dist.standard_normal_cdf(np.log(a / rf_r) / b - 0.5 * b)
    sigmas = [0.01 + k * 0.01 for k in range(100)]
    mp = pu.PlotUtilities('Probability of Exceeding Goal', '$b$', 'Probability')
    print(limit_prob)
    mp.multi_plot(sigmas, [[dist.standard_normal_cdf(np.log(a / rf_r) / s - 0.5 * s) for s in sigmas]])

    pis = [0.05 * (k+1) for k in range(500)]
    probs = []
    for pi in pis:
        p = dist.standard_normal_cdf(- 0.5 * b - np.log((rf_r + (_x0/_x - rf_r)/pi) / a) / b)
        probs.append(p)

    mp = pu.PlotUtilities('Goal Reaching Probability', '$\pi$', 'Probability')
    mp.multi_plot(pis, [probs])

    _pi = 0.6
    exact_prob = dist.standard_normal_cdf(- 0.5 * b - np.log((rf_r + (_x0 / _x - rf_r) / _pi) / a) / b)
    print(exact_prob)

    n_repeats = 1000
    sample_size = 5000
    sample_res_p_meas = []
    for k in range(n_repeats):
        ns = np.random.normal(0, 1, sample_size)
        sample_out = [1. if _x * (rf_r + _pi * (a * np.exp(b * x - 0.5 * b * b) - rf_r)) > _x0 else 0. for x in ns]
        sample_res_p_meas.append(sum(sample_out) / sample_size)

    mp = pu.PlotUtilities('Histogram Reaching Probability - P-measure', '$\pi$', 'Occurrence')
    mp.plot_histogram(sample_res_p_meas, 20)

    print('Sample Variance : ' + str(np.var(np.array(sample_res_p_meas))))

    shifts = [0.05 * k for k in range(50)]
    variance_by_shift = []
    all_ns = []
    all_portfolio_values = []
    for k in range(n_repeats):
        ns = np.random.normal(0, 1, sample_size)
        all_ns.append(ns)
        pf_single = [1. if _x * (rf_r + _pi * (a * np.exp(b * x - 0.5 * b * b) - rf_r)) > _x0 else 0. for x in ns]
        all_portfolio_values.append(pf_single)

    print('All Portfolio Values assembled.')

    for s in shifts:
        sample_res_q = []
        for k in range(n_repeats):
            payoff = 0.
            for (x, pv) in zip(all_ns[k], all_portfolio_values[k]):
                # payoff = payoff + pv
                payoff = payoff + np.exp(-s * x + 0.5 * s * s) * pv
            sample_res_q.append(payoff / sample_size)
        variance_by_shift.append(np.var(sample_res_q))

    mp = pu.PlotUtilities('Sample Variance By Shift', '$Shift$', 'Occurrence')
    mp.multi_plot(shifts, [variance_by_shift])

    print('Optimal Shift: ' + str(shifts[np.argmin(variance_by_shift)]))

    s_star = shifts[np.argmin(variance_by_shift)]
    # s_star = 0.95
    sample_res_p_meas = []
    for k in range(n_repeats):
        ns = np.random.normal(0, 1, sample_size)
        sample_out = [np.exp(-s_star * (x + s_star) + 0.5 * s_star * s_star)
                      if _x * (rf_r + _pi * (a * np.exp(b * (x + s_star) - 0.5 * b * b) - rf_r)) > _x0 else 0.
                      for x in ns]
        sample_res_p_meas.append(sum(sample_out) / sample_size)

    mp = pu.PlotUtilities('Histogram Reaching Probability - Shifted Measure', '$\pi$', 'None')
    mp.plot_histogram(sample_res_p_meas, 20)

    print('Sample Variance : ' + str(np.var(np.array(sample_res_p_meas))))
