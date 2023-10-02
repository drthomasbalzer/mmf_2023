"""
Author: Thomas Balzer
(c) 2023
Material for MMF Stochastic Analysis - Fall 2023
"""


import numpy as np

import core_math_utilities as dist
import plot_utilities as pu


def exponential_importance_sampling(_lambda, threshold, shift):

    """
    we evaluate a payout of either $P(X > t)$ for an exponential distribution
    with a given _lambda and a possible shift to apply to the parameter
    """

    repeats = 1000

    sample_base = []  # this is the sample WITHOUT I.S. applied
    sample_is = []  # this is the sample WITH I.S. applied

    for z in range(repeats):

        # we are sampling sz times in each iteration
        sz = 5000

        lower_bound = 0.
        upper_bound = 1.

        # we create a uniform sample first;
        uni_sample = np.random.uniform(lower_bound, upper_bound, sz)

        # transform the uniform sample to exponentials with two different shifts
        sample_exp = [dist.exponential_inverse_cdf(_lambda, u) for u in uni_sample]
        sample_exp_shift = [dist.exponential_inverse_cdf(_lambda + shift, u) for u in uni_sample]

        # evaluate the payout

        c = (_lambda + shift) / _lambda

        payout_base = sum([1 if se > threshold else 0. for se in sample_exp])
        payout_is = sum([np.exp(shift * ses) / c * (1 if ses > threshold else 0.) for ses in sample_exp_shift])

        sample_base.append(payout_base / sz)
        sample_is.append(payout_is / sz)

    # prepare and show plot

    num_bins = 50

    # this is the exact result

    print(np.var(sample_base))
    print(np.var(sample_is))

    colors = ['green', 'blue']
    _title = "Tail Probability with and w/o Importance Sampling"

    mp = pu.PlotUtilities(_title, 'Outcome', 'Rel. Occurrence')
    mp.plot_multi_histogram([sample_base, sample_is], num_bins, colors)


if __name__ == '__main__':

    _intensity = 0.275
    print('Expected Value - ' + str(1. / _intensity))
    _t = 15
    _shift = 1. / _t - _intensity
    print(_shift)
    print('Expected Value (Shifted) - ' + str(1. / (_intensity + _shift)))
    #
    # _shift = 1. / _t - np.sqrt(1. / (_t * _t) + _intensity * _intensity)
    # print(_shift)
    # print('Expected Value (Shifted) - ' + str(1. / (_intensity + _shift)))
    exponential_importance_sampling(_intensity, _t, _shift)
