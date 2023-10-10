"""
Author: Thomas Balzer
(c) 2023
Material for MMF Stochastic Analysis - Fall 2023
"""

import numpy as np
import plot_utilities as pu
import core_math_utilities as dist


"""
Option Pricing With Default Risk - Homework #6
"""


def _d1(s, threshold, sigma):

    return 1./sigma * np.log(threshold / s) + 0.5 * sigma


def _h(z, rho, a, d1):

    return (1. if z > d1 else 0.) * dist.standard_normal_cdf((rho * z - a) / np.sqrt(1 - rho * rho))


if __name__ == '__main__':

    print('Option Pricing With Default Risk - Homework #6')

    _pd = 0.05
    _sigma = 0.2
    _K = 1.25
    _s = 1.
    '''
    Modelling Default through a Normal Random Variable with 
    $\{ \tau < t \} = \{ Y < a \}$ obviously means that 
    a = \Phi^{-1}(p(t)) needs to hold
    '''

    _a = dist.normal_cdf_inverse(_pd)
    print('Default Threshold: ' + str(_a))

    d1_v = _d1(_s, _K, _sigma)

    '''
    Calculation of Riskfree Value
    '''
    v_0 = 1 - dist.standard_normal_cdf(d1_v)
    print('Riskfree Value: ' + str(v_0))

    '''
    Calculation of Risk-Adjusted Value (Independent)
    '''
    print('Independent Risk-Adjusted Value: ' + str((1-_pd) * v_0))

    x_label = 'z'
    y_label = 'h(z)'
    chart_title = 'h(z) as function of z'

    z_ax = [-2 + 0.01 * k for k in range(400)]
    _rho = 0.5
    y_ax = [_h(z, _rho, _a, d1_v) for z in z_ax]
    mp = pu.PlotUtilities(chart_title, x_label, y_label)
    mp.multi_plot(z_ax, [y_ax])

    _sample_size = 10000
    normal_sample = np.random.normal(0., 1., _sample_size)

    rhos = [-0.99 + k * 0.01 for k in range(199)]
    v = []
    for r in rhos:
        h_v = [_h(z, r, _a, d1_v) for z in normal_sample]
        v.append(sum(h_v) / _sample_size)

    x_label = 'Correlation'
    y_label = 'Risk-Adjusted Value'
    chart_title = 'Value as a Function of Correlation'

    mp = pu.PlotUtilities(chart_title, x_label, y_label)
    mp.multi_plot(rhos, [v])
