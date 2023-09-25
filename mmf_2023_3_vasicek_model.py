"""
Author: Thomas Balzer
(c) 2023
Material for MMF Stochastic Analysis - Fall 2023
"""

import numpy as np

import core_math_utilities as dist
import plot_utilities as pu


# Demo of Vasicek Model

def vasicek_large_portfolio_cdf(rho, p, x):

    nd = dist.NormalDistribution(0., 1.)
    y = (nd.inverse_cdf(x) * np.sqrt(1 - rho) - nd.inverse_cdf(p)) / np.sqrt(rho)
    return nd.cdf(y)


def plot_vasicek_distribution(rhos, p, min_val, max_val, steps):

    x_label = 'x'
    y_label = 'CDF Value'
    chart_title = 'Vasicek Large Portfolio Distribution For Different Correlations'
    step = (max_val - min_val) / steps
    x_ax = [min_val + step * k for k in range(steps)]
    y_ax = [[vasicek_large_portfolio_cdf(rho, p, x_val) for x_val in x_ax] for rho in rhos]

    mp = pu.PlotUtilities(chart_title, x_label, y_label)
    mp.multi_plot(x_ax, y_ax)


def conditional_default_prob(rho, p, z):
    nd = dist.NormalDistribution(0., 1.)
    y = (nd.inverse_cdf(p) - np.sqrt(rho) * z) / np.sqrt(1 - rho)
    return nd.cdf(y)


class FunctorConditionalLossDist:

    def __init__(self, rho, p, k, n, use_poisson=False):

        self.p = p
        self.rho = rho
        self.k = k
        self.N = n
        self.usePoissonApprox = use_poisson

    def pdf(self, z):

        p_z = conditional_default_prob(self.rho, self.p, z)
        if self.usePoissonApprox:
            return dist.poisson_pdf(self.N * p_z, self.k)
        else:
            return dist.binomial_pdf(p_z, self.k, self.N)


class FunctorConditionalFirstLoss:

    def __init__(self, rho, p, n, loss_n):

        self.p = p
        self.rho = rho
        self.N = n
        self.loss_n = int(loss_n)

    def first_loss(self, z):

        p_z = conditional_default_prob(self.rho, self.p, z)
        fl = self.loss_n
        for k in range(self.loss_n):
            fl = fl + (k - self.loss_n) * dist.binomial_pdf(p_z, k, self.N)
        return fl


def portfolio_loss_histogram(rho, p, n, plot_vs_lhp=False):

    x_label = 'x'
    y_label = 'CDF Value'
    chart_title = 'Portfolio Loss Distribution'
    x_ax = [k for k in range(n+1)]
    pdf_v = [0.] * (n+1)

    for k in range(n+1):
        pdf_func = FunctorConditionalLossDist(rho, p, k, n, False)
        pdf_v[k] = gauss_hermite_integration_normalised(20, pdf_func.pdf)

    cdf_v = [sum([pdf_v[i] for i in range(k)]) for k in range(n+1)]

    all_values = [cdf_v]
    labels = ['Discrete CDF']
    colors = ['blue']
    if plot_vs_lhp:
        normed_ax = [float(x) / float(n) for x in x_ax]
        lp_val = [vasicek_large_portfolio_cdf(rho, p, x) for x in normed_ax]
        all_values.append(lp_val)
        labels.append('Vasicek CDF')
        colors.append('green')

    mp = pu.PlotUtilities(chart_title, x_label, y_label)
    mp.sub_plots(x_ax, all_values, labels, colors)


def gauss_hermite_integration_normalised(deg, func):

    x, y = np.polynomial.hermite.hermgauss(deg)

    scaling_factor = 1./np.sqrt(np.pi)
    absc_factor = np.sqrt(2.)
    val = sum([func(absc_factor * x_i) * y_i for x_i, y_i in zip(x, y)]) * scaling_factor
    return val


def example_bivariate_option_price_mc(mean, variance, pd, pd_vol, strike, df):
    size = 15000
    standard_normal_sample = np.random.standard_normal(2 * size)
    # we turn this into a random sample of dimension 2;

    vol = np.sqrt(variance)
    x = [mean * np.exp(vol * standard_normal_sample[k] - 0.5 * variance) for k in range(size)]
    y = [0.] * size

    option_value = df * sum([max(x_0 - strike, 0) for x_0 in x]) / size
    print(option_value)
    min_rho = -0.99
    max_rho = 0.99
    step_size = 0.01
    rho_steps = int((max_rho - min_rho) / step_size)
    rhos = [min_rho + k * step_size for k in range(rho_steps)]
    option_values = [option_value] * len(rhos)

    mc_value = []
    default_threshold = dist.normal_cdf_inverse(pd) * pd_vol
    for rho in rhos:
        for k in range(size):
            z = pd_vol * (rho * standard_normal_sample[k] + np.sqrt(1 - rho * rho) * standard_normal_sample[k + size])
            y[k] = (0. if z <= default_threshold else 1.)
        mc_value.append(df * sum([y_0 * max(x_0 - strike, 0.) for y_0, x_0 in zip(y, x)]) / size)

    mp = pu.PlotUtilities('Risk-Adjusted Option Value As Function of Correlation', 'Correlation', 'Option Value')
    mp.multi_plot(rhos, [option_values, mc_value])


if __name__ == '__main__':

    _size = 100
    _rhos = [0.05, 0.1, 0.2, 0.5, 0.75]

    # -- portfolio loss distribution for finite case
    _p = 0.05
    portfolio_loss_histogram(_rhos[2], _p, _size, False)
    portfolio_loss_histogram(_rhos[2], _p, _size, True)

    # -- portfolio loss distribution for LHP case
    plot_vasicek_distribution(_rhos, _p, 0., 0.25, 500)
