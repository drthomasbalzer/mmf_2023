"""
Author: Thomas Balzer
(c) 2023
Material for MMF Stochastic Analysis - Fall 2023
"""


from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

import core_math_utilities as dist


def transition_density(_x, _y, _t):

    return dist.normal_pdf(_x, _y, _t)


def expected_positive_exposure(_mean, _variance):
    y = _mean / np.sqrt(_variance)
    return _mean * dist.standard_normal_cdf(y) + np.sqrt(_variance) * dist.standard_normal_pdf(y)


def excess_probability_payoff(_strike, _mean, _variance):
    return 1 - dist.standard_normal_cdf((_strike - _mean) / np.sqrt(_variance))


def second_moment(_mean, _variance):
    return _mean * _mean + _variance


def mgf(_mean, _variance):

    return np.exp(_mean + _variance * 0.5)


def plot_transition_functor(option):

    # set up the axes for the first plot
    z_lim = 1.0

    # preparation of charts
    if option == 0 or option == 1:
        x = np.arange(0.1, 3, 0.2)
        y = np.arange(-3, 3, 0.1)
    elif option == 2 or option == 3:
        x = np.arange(0.05, 3, 0.1)
        y = np.arange(-2, 2, 0.1)
    elif option == 4:
        x = np.arange(0.05, 1.5, 0.1)
        y = np.arange(-1, 1.5, 0.1)
        z_lim = 3.
    else:
        raise Exception('Invalid Calc Option.')

    x, y = np.meshgrid(x, y)

    """
    set up x, y values and populate function values
    ~~ calculation options
    0 - transition probability
    1 - second moment
    2 - excess probability (digital option payout)
    3 - expected positive exposure
    """

    if option == 0:
        z = transition_density(0, y, x)
    elif option == 1:
        z = second_moment(y, x)
        z_lim = 10
    elif option == 2:
        z = excess_probability_payoff(-0.5, y, x)
    elif option == 3:
        z = expected_positive_exposure(y, x)
    elif option == 4:
        z = mgf(y, x)
    else:
        raise Exception('Invalid Calc Option.')

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='coolwarm',
                           linewidth=0, antialiased=True)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    ax.set_zlim(0, z_lim)
    fig.colorbar(surf, shrink=0.75, aspect=8)

    plt.show()


if __name__ == '__main__':

    for k in range(5):
        plot_transition_functor(k)
