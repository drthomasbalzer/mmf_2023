"""
Author: Thomas Balzer
(c) 2023
Material for MMF Stochastic Analysis - Fall 2023
"""

import numpy as np
import plot_utilities as pu
import core_math_utilities as dist


def bivariate_payoff(s_z, s_w, rho, def_th, d1):

    ind_1 = (1. if rho * s_z + np.sqrt(1-rho*rho) * s_w > def_th else 0.)
    ind_2 = (1. if s_z > d1 else 0.)
    return ind_1 * ind_2


def gauss_integration_weights_simple(_deg):

    # alternative approach - more basic integration from -5 to +5
    _absc = [-5. + k * 10. / _deg for k in range(_deg)]
    _weights = []
    w_prev = dist.standard_normal_cdf(_absc[0])
    _weights.append(w_prev)
    for a in _absc[1:]:
        w = dist.standard_normal_cdf(a)
        _weights.append(w - w_prev)
        w_prev = w

    return _absc, _weights


"""
Option Pricing With Default Risk - Assignment
"""


def _d1(s, threshold, sigma, shift=0.):

    return 1./sigma * np.log(threshold / s) + 0.5 * sigma - shift


def _h(z, rho, a, d1):

    return (1. if z > d1 else 0.) * dist.standard_normal_cdf((rho * z - a) / np.sqrt(1 - rho * rho))


def _dh_drho(z, rho, a, d1):

    m_rsq = np.sqrt(1 - rho * rho)
    return ((1. if z > d1 else 0.) *
            dist.standard_normal_pdf((rho * z - a) / m_rsq) * (z + (z * rho - a) * rho / m_rsq / m_rsq) / m_rsq)


if __name__ == '__main__':

    print('Option Pricing With Default Risk - Assigment #1')

    _pd = 0.05
    _sigma = 0.2
    _K = 1.25
    _s = 1.

    ''' 
    Calculation of risk-adjusted price as a function of $p$ in independent case
    '''
    d1_v = _d1(_s, _K, _sigma)
    v_0 = 1 - dist.standard_normal_cdf(d1_v)
    print('Riskfree Value: ' + str(v_0))

    probs = [0.002 * k for k in range(100)]
    mp = pu.PlotUtilities('Risk-Adjusted Value - Independent Case', 'Default Probability', 'Risk-Adjusted Price')
    mp.multi_plot(probs, [[v_0 * (1-p) for p in probs]])

    '''
    Case 2 - Correlation of -1 
    '''
    values = []
    for p in probs:
        this_threshold = - dist.normal_cdf_inverse(p)
        if this_threshold < d1_v:
            values.append(0.)
        else:
            values.append(dist.standard_normal_cdf(this_threshold) - dist.standard_normal_cdf(d1_v))

    mp = pu.PlotUtilities('Risk-Adjusted Value - Correlation = -1', 'Default Probability', 'Risk-Adjusted Price')
    mp.multi_plot(probs, [values])

    '''
    brute-force calculation as a function of $\rho$
    '''

    _sample_size = 10000
    random_sample_z = np.random.normal(0., 1., _sample_size)
    random_sample_w = np.random.normal(0., 1., _sample_size)

    rho_range = [-1. + (k+1) * 0.005 for k in range(398)]
    v_1 = []
    default_threshold = dist.normal_cdf_inverse(_pd)
    print(default_threshold)
    v_r = []
    for r in rho_range:
        sample = [bivariate_payoff(s_z, s_w, r, default_threshold, d1_v)
                  for (s_z, s_w) in zip(random_sample_z, random_sample_w)]
        v_r.append(sum(sample) / _sample_size)

    mp = pu.PlotUtilities('Brute-Force Simulation of $Z$ and $W$', 'Correlation', 'Value')
    mp.multi_plot(rho_range, [v_r])

    correlation_bump = 1.e-06
    v_corr_risk_br = [0]
    index = 1
    for r in rho_range[1:]:
        v_corr_risk_br.append((v_r[index] - v_r[index-1])/(rho_range[index] - rho_range[index-1]))
        index = index + 1

    mp = pu.PlotUtilities('Brute-Force Simulation of $Z$ and $W$ - Correlation Risk', 'Correlation', 'Value')
    mp.sub_plots(rho_range, [v_r, v_corr_risk_br], ['Value', 'Risk'], ['red', 'green'])

    '''
    Simulation of $Z$ only
    '''
    v_r = []
    for r in rho_range:
        sample = [_h(s_z, r, default_threshold, d1_v) for s_z in random_sample_z]
        v_r.append(sum(sample) / _sample_size)

    mp = pu.PlotUtilities('Simulation of $Z$ Only', 'Correlation', 'Value')
    mp.multi_plot(rho_range, [v_r])

    v_corr_risk_analytic = []
    for r in rho_range:
        sample = [_dh_drho(s_z, r, default_threshold, d1_v) for s_z in random_sample_z]
        v_corr_risk_analytic.append(sum(sample) / _sample_size)

    mp = pu.PlotUtilities('Simulation of $Z$ Only - Correlation Risk', 'Correlation', 'Value')
    mp.sub_plots(rho_range, [v_r, v_corr_risk_analytic], ['Value', 'Risk'], ['red', 'green'])

    ''' 
    Numerical Integration over $Z$ 
    '''
    v_r = []
    v_r_p = []
    v_corr_risk_analytic = []
    n_deg = 100
    x, y = gauss_integration_weights_simple(n_deg)
    for r in rho_range:
        sample = 0.
        sample_p = 0.
        sample_risk = 0.
        for k in range(n_deg):
            sample = sample + _h(x[k], r, default_threshold, d1_v) * y[k]
            sample_p = sample_p + _h(x[k], r + correlation_bump, default_threshold, d1_v) * y[k]
            sample_risk = sample_risk + _dh_drho(x[k], r, default_threshold, d1_v) * y[k]

        v_r.append(sample)
        v_r_p.append(sample_p)
        v_corr_risk_analytic.append(sample_risk)

    v_corr_risk_br = [(v_rp_v - v_r_v) / correlation_bump for (v_rp_v, v_r_v) in zip(v_r_p, v_r)]

    mp = pu.PlotUtilities('Numerical Integration over $Z$', 'Correlation', 'Value')
    mp.sub_plots(rho_range, [v_r, v_corr_risk_analytic, v_corr_risk_br],
                 ['Value', 'Risk', 'Risk B&R'], ['red', 'green', 'blue'])
