"""
Author: Thomas Balzer
(c) 2023
Material for MMF Stochastic Analysis - Fall 2023
"""

import plot_utilities as pu
import core_math_utilities as dist

"""
set up plots for pdf of multiple distributions
"""


def plot_multi_distributions(distrib, min_v, max_v, steps, chart_title):

    x_label = 'x'
    y_label = 'PDF Value'

    step = (max_v - min_v) / steps
    x_ax = [min_v + step * k for k in range(steps)]
    y_ax = [[dt.pdf(x_val) for x_val in x_ax] for dt in distrib]

    mp = pu.PlotUtilities(chart_title, x_label, y_label)
    mp.multi_plot(x_ax, y_ax)


def plot_pdf_v_cdf(distrib, min_v, max_v, steps, chart_title):

    step = (max_v - min_v) / steps
    x_ax = [min_v + step * k for k in range(steps)]
    pdf_value = [distrib.pdf(x_val) for x_val in x_ax]
    cdf_value = [distrib.cdf(x_val) for x_val in x_ax]

    mp = pu.PlotUtilities(chart_title, 'x', 'unused')
    mp.sub_plots(x_ax, [pdf_value, cdf_value], ['PDF', 'CDF'], ['blue', 'red'])


if __name__ == '__main__':

    n_plots = 5

    versions = [k for k in range(1, 5)]

    plotCDF = False

    for version in versions:
        title = ''
        if version == 1:
            # example of normal distributions with different $\sigma$ and $\mu = 0$.
            min_val = -5.
            max_val = 5.
            title = 'Normal Distribution with fixed $\mu$'
            d = [dist.NormalDistribution(0., 1. * (1. + float(k))) for k in range(n_plots)]
        elif version == 2:
            # example of normal distributions with different $\mu$ and $\sigma = 1$.
            min_val = -5.
            max_val = 5.
            title = 'Normal Distribution with fixed $\sigma$'
            d = [dist.NormalDistribution(-2. + float(k), 1.) for k in range(n_plots)]
        elif version == 3:
            # example of Exponential distributions
            min_val = 0.
            max_val = 10.
            title = 'Exponential Distribution'
            d = [dist.ExponentialDistribution(0.5 * (1. + float(k))) for k in range(n_plots)]
        elif version == 4:
            # Example of Uniform Distributions with different ranges of values
            min_val = -1.
            max_val = n_plots + 1
            title = 'Uniform Distribution'
            d = [dist.UniformDistribution(0., 1. + float(k)) for k in range(n_plots)]
        else:
            print('Unsupported Distribution Choice.')

        if plotCDF:
            plot_pdf_v_cdf(d[0], min_val, max_val, 1000, title)
        else:
            plot_multi_distributions(d, min_val, max_val, 1000, title)
