"""
Author: Thomas Balzer
(c) 2023
Material for MMF Stochastic Analysis - Fall 2023
"""

import plot_utilities as pu
import mmf_2023_3_vasicek_model as vm
import numpy as np
import core_math_utilities as dist

"""
Portfolio Loss Modelling - Homework #2
"""


def portfolio_first_loss_full(p, n, _loss_percentage, _size):

    # we need $n+1$ independent random variables, _size times
    # we will be interpreting the first $n$ as the idiosyncratic variables and the last as systematic
    normal_samples = []
    for k in range(_size):
        normal_samples.append(np.random.normal(0., 1., n+1))

    default_threshold = dist.normal_cdf_inverse(p)
    rhos = [0.01 * k for k in range(99)]
    first_loss = []
    for rho in rhos:
        fl = 0.
        sqrt_rho = np.sqrt(rho)
        one_m_sqrt_rho = np.sqrt(1-rho)
        for ns in normal_samples:
            z = ns[n]
            tl = 0.
            for k in range(n):
                tl = tl + (1./n if sqrt_rho * z + one_m_sqrt_rho * ns[k] < default_threshold else 0.)
            fl = fl + min(tl, _loss_percentage)
        first_loss.append(fl/_size)

    x_label = 'rho'
    y_label = 'First Loss'
    chart_title = 'First Loss as Function of Rho'

    mp = pu.PlotUtilities(chart_title, x_label, y_label)
    mp.multi_plot(rhos, [first_loss])


def portfolio_first_loss_semi_analytic(p, n, _loss_percentage, _size):

    # we need only _size independent random variables since we are only simulating the systematic risk
    normal_sample = np.random.normal(0., 1., _size)

    rhos = [0.01 * k for k in range(99)]
    first_loss = []
    for rho in rhos:
        fl_total = 0
        for z in normal_sample:
            f = vm.FunctorConditionalFirstLoss(rho, p, n, int(_loss_percentage * n))
            fl_total = fl_total + f.first_loss(z) / n
        first_loss.append(fl_total / _size)

    x_label = 'rho'
    y_label = 'First Loss Semi-Analytic'
    chart_title = 'First Loss (Semi-Analytic) as Function of Rho'

    mp = pu.PlotUtilities(chart_title, x_label, y_label)
    mp.multi_plot(rhos, [first_loss])


def portfolio_first_loss_vasicek(p, _loss_percentage, _size):

    # we need only _size independent random variables since we are only simulating the systematic risk
    normal_sample = np.random.normal(0., 1., _size)

    rhos = [0.01 * k for k in range(99)]
    first_loss = []
    for rho in rhos:
        fl_total = 0
        for z in normal_sample:
            p_z = vm.conditional_default_prob(rho, p, z)
            fl_total = fl_total + min(p_z, _loss_percentage)
        first_loss.append(fl_total / _size)

    x_label = 'rho'
    y_label = 'First Loss Vasicek LHP'
    chart_title = 'First Loss (Vasicek) as Function of Rho'

    mp = pu.PlotUtilities(chart_title, x_label, y_label)
    mp.multi_plot(rhos, [first_loss])


def portfolio_first_loss_gauss_hermite(p, n, _loss_percentage, deg):

    # in this case we are performing Gauss Hermite integration with deg degrees of freedom
    rhos = [0.01 * k for k in range(99)]
    first_loss = []
    for rho in rhos:
        f = vm.FunctorConditionalFirstLoss(rho, p, n, _loss_percentage * n)
        first_loss.append(vm.gauss_hermite_integration_normalised(deg, f.first_loss) / n)

    x_label = 'rho'
    y_label = 'First Loss Gauss-Hermite'
    chart_title = 'First Loss (Gauss-Hermite) as Function of Rho'

    mp = pu.PlotUtilities(chart_title, x_label, y_label)
    mp.multi_plot(rhos, [first_loss])


if __name__ == '__main__':

    # basic parameters applicable to all questions
    _p = 0.05
    _N = 20
    _loss_pc = 0.05
    mc_size = 5000

    portfolio_first_loss_full(_p, _N, _loss_pc, mc_size)

    portfolio_first_loss_semi_analytic(_p, _N, _loss_pc, mc_size)

    _deg = 31
    portfolio_first_loss_gauss_hermite(_p, _N, _loss_pc, _deg)

    portfolio_first_loss_vasicek(_p, _loss_pc, mc_size)
