"""
Author: Thomas Balzer
(c) 2023
Material for MMF Stochastic Analysis - Fall 2023
"""


import numpy as np

import core_math_utilities as dist
import plot_utilities as pu
import matplotlib.pyplot as plt

"""
Demo of the Law of Large Numbers
"""


def binomial_lln(sample_size, p):

    # Step 1 - create sample of independent uniform random variables
    uni_sample = np.random.uniform(0., 1., sample_size)

    # Step 2 - transform them to $B(1,p)$ distribution
    sample = [dist.binomial_inverse_cdf(p, u) for u in uni_sample]

    x_ax = [k for k in range(sample_size)]  # values on the x axis

    # y_values (1) - actual average
    y_ax_1 = [p] * sample_size

    # y_values (0) - cumulative average of all the samples
    y_ax_0 = [sum(sample[0:k+1]) / (k+1) for k in range(sample_size)]

    mp = pu.PlotUtilities("Cumulative Average", 'x', 'Average')
    mp.multi_plot(x_ax, [y_ax_0, y_ax_1])


def binomial_lln_hist(sample_size, repeats, p):

    # plot histogram of Average value of normalised Binomial Distributions

    sample_value = [0.] * repeats
    num_bins = 35
    for i in range(repeats):

        # Step 1 - create sample of independent uniform random variables

        lower_bound = 0.
        upper_bound = 1.
        uni_sample = np.random.uniform(lower_bound, upper_bound, sample_size)

        # Step 2 - transform them to $B(1,p)$ distribution

        sample = [dist.binomial_inverse_cdf(p, u) for u in uni_sample]
        sample_value[i] = (sum(sample) - sample_size * p) / np.sqrt(p * (1-p)) / sample_size

    mp = pu.PlotUtilities("Histogram of Normalised Binomial Average For Sample of Size={0}".format(sample_size),
                          'Outcome', 'Rel. Occurrence')
    mp.plot_histogram(sample_value, num_bins)


def binomial_clt_hist(sample_size, repeats, p):

    # plot histogram of Average value of normalised Binomial Distributions

    sample_value = [0.] * repeats
    num_bins = 35
    for i in range(repeats):

        # Step 1 - create sample of independent uniform random variables

        lower_bound = 0.
        upper_bound = 1.
        uni_sample = np.random.uniform(lower_bound, upper_bound, sample_size)

        # Step 2 - transform them to $B(1,p)$ distribution

        sample = [dist.binomial_inverse_cdf(p, u) for u in uni_sample]
        sample_value[i] = (sum(sample) - sample_size * p) / np.sqrt(sample_size * p * (1-p))

    plt.xlabel('Outcome')
    plt.ylabel('Relative Occurrence')
    plt.title("Histogram of Normalised Binomial Average For Sample of Size={0}".format(sample_size))

    _n, bins, _hist = plt.hist(sample_value, num_bins, density=True, facecolor='green', alpha=0.75)

    y = [dist.standard_normal_pdf(b) for b in bins]
    plt.plot(bins, y, 'r--')
    plt.show()
    plt.close()


if __name__ == '__main__':

    _sz = 1000
    _p = .75
    binomial_lln(_sz, _p)
    _repeats = 10000
    binomial_lln_hist(_sz, _repeats, _p)
    binomial_clt_hist(_sz, _repeats, _p)
