"""
Author: Thomas Balzer
(c) 2023
Material for MMF Stochastic Analysis - Fall 2023
"""

import numpy as np
import core_math_utilities as dist
import plot_utilities as pu


def correlated_defaults_scatter(lambda_1, lambda_2, rhos, sample_size):

    tau_2 = [0] * len(rhos)

    sns = np.random.standard_normal(2 * sample_size)

    x = sns[:sample_size]
    tau_1 = [dist.exponential_inverse_cdf(lambda_1, dist.standard_normal_cdf(x1)) for x1 in x]

    index = 0
    for rho in rhos:
        y = [rho * sns[k] + np.sqrt(1 - rho * rho) * sns[k + sample_size] for k in range(sample_size)]
        tau_2[index] = [dist.exponential_inverse_cdf(lambda_2, dist.standard_normal_cdf(y1)) for y1 in y]
        index = index + 1

    # scatter plot of the simulated defaults
    colors = ['blue', 'green', 'orange', 'red', 'yellow']

    mp = pu.PlotUtilities('Default Times with Correlations={0}'.format(rhos), 'x', 'y')

    mp.scatter_plot(tau_1, tau_2, rhos, colors)


def vasicek_large_portfolio_cdf(rho, p, x):

    nd = dist.NormalDistribution(0., 1.)
    y = (nd.inverse_cdf(x) * np.sqrt(1 - rho) - nd.inverse_cdf(p)) / np.sqrt(rho)
    return nd.cdf(y)


def plot_vasicek_distribution(rhos, p, min_val, max_val, steps):

    x_label = 'x'
    y_label = 'CDF Value'
    chart_title = 'Vasicek Large Portfolio Distribution'
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


def gauss_hermite_integration_normalised(deg, func):

    x, y = np.polynomial.hermite.hermgauss(deg)

    scaling_factor = 1./np.sqrt(np.pi)
    absc_factor = np.sqrt(2.)
    val = sum([func(absc_factor * x_i) * y_i for x_i, y_i in zip(x, y)]) * scaling_factor
    return val


if __name__ == '__main__':

    """
    demo of simple correlated defaults with different correlations
    """
    _sample_size = 100
    _lambda_1 = 0.25
    _lambda_2 = 0.5
    _rhos = [0.95, 0.5, 0., -0.5, -0.95]
    correlated_defaults_scatter(_lambda_1, _lambda_2, _rhos, _sample_size)
