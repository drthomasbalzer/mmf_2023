"""
Author: Thomas Balzer
(c) 2023
Material for MMF Stochastic Analysis - Fall 2023
"""


import numpy as np
import math


class Distribution:

    def pdf(self, x):
        return 0.

    def cdf(self, x):
        return 0.

    def inverse_cdf(self, x):
        return 0.

    def excess_probability(self, strike):
        return 1 - self.cdf(strike)


class NormalDistribution(Distribution):

    def __init__(self, mu, sigma_sq):

        self.mu = mu
        self.sigma_sq = sigma_sq

    def pdf(self, x):
        return normal_pdf(x, self.mu, self.sigma_sq)

    def cdf(self, x):
        # P(X \leq x) = P(sigma * X_0 + mean \leq x) = P(X_0 \leq (x - mean)/sigma)
        return standard_normal_cdf((x - self.mu) / np.sqrt(self.sigma_sq))

    def expected_positive_exposure(self):
        return self.call_option(0.)

    def call_option(self, strike):
        y = (strike - self.mu) / np.sqrt(self.sigma_sq)
        return (self.mu - strike) * (1. - self.cdf(strike)) + np.sqrt(self.sigma_sq) * standard_normal_pdf(y)

    def put_option(self, strike):
        return self.call_option(strike) + strike - self.mu

    def second_moment(self):
        return self.mu * self.mu + self.sigma_sq

    def inverse_cdf(self, x):
        return self.mu + np.sqrt(self.sigma_sq) * standard_normal_inverse_cdf(x)


class UniformDistribution(Distribution):

    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def pdf(self, x):
        if x < self.lower or x > self.upper:
            return 0.
        else:
            return 1. / (self.upper - self.lower)

    def cdf(self, x):
        if x < self.lower:
            return 0.
        elif x > self.upper:
            return 1.
        else:
            return (x - self.lower) / (self.upper - self.lower)


class ExponentialDistribution(Distribution):

    def __init__(self, _lambda):
        self._lambda = _lambda

    def pdf(self, x):
        return exponential_pdf(self._lambda, x)

    def cdf(self, x):
        if x < 0.:
            return 0.
        else:
            return 1 - np.exp(-self._lambda * x)

    def inverse_cdf(self, x):
        return exponential_inverse_cdf(self._lambda, x)


"""
PDF of normal distribution
"""


def normal_pdf(x, mu, sigma_sq):
    y = 1. / np.sqrt(2 * np.pi * sigma_sq) * np.exp(- 0.5 * np.power(x - mu, 2) / sigma_sq)
    return y


def standard_normal_pdf(x):
    return normal_pdf(x, 0, 1)


def standard_normal_cdf(x):
    return 0.5 * (1.0 + erf(x / np.sqrt(2)))


def rational_approximation(t):
    # Abramowitz and Stegun formula 26.2.23.
    # The absolute value of the error should be less than 4.5 e-4.
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    numerator = (c[2] * t + c[1]) * t + c[0]
    denominator = ((d[2] * t + d[1]) * t + d[0]) * t + 1.0
    return t - numerator / denominator

def normal_cdf_inverse(p):

    # some judicious choices for $p$ outside of the relevant area
    if p >= 1:
        return 10000.
    elif p <= 0:
        return -10000.

    # See article above for explanation of this section.
    if p < 0.5:
        # F^-1(p) = - G^-1(p)
        return -rational_approximation(np.sqrt(-2.0 * np.log(p)))
    else:
        # F^-1(p) = G^-1(1-p)
        return rational_approximation(np.sqrt(-2.0 * np.log(1.0 - p)))


###################################################################################################

def erf(x):
    # constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    # Save the sign of x
    sign = np.sign(x)
    x = np.abs(x)

    # Formula 7.1.26 given in Abramowitz and Stegun.
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)

    return sign * y


####################################################################################################



def standard_normal_inverse_cdf(x):

    return normal_cdf_inverse(x)

"""
PDF of Maximum of Brownian Motion
"""


def brownian_running_max_pdf(x, _time):
    return 2 / np.sqrt(_time) * standard_normal_pdf(- x / np.sqrt(_time))
    # * np.sqrt(2 / (np.pi * _time)) * np.exp( - 0.5 * x * x / _time)


"""
Exponential distribution with parameter $\lambda > 0$
-- pdf and inverse distribution function of $Exp(\lambda)$-distribution
"""


def exponential_pdf(_lambda, x):

    if x < 0:
        return 0.
    else:
        return _lambda * np.exp(-_lambda * x)


def exponential_inverse_cdf(_lambda, x):

    if x < 0:
        return 0.
    else:
        return -1./_lambda * np.log(1-x)


"""
Details of a random variable $X$ with $\mathbb{P}(X = 1) = p = 1 - \mathbb{P}(X = 0)$
"""


def binomial_inverse_cdf(p, x):

    if x <= 1 - p:
        return 0.
    else:
        return 1.


def binomial_expectation(p):
    return p


def binomial_variance(p):
    return p * (1 - p)


"""
Details of a random variable $X$ with $\mathbb{P}(X = 1) = p = 1 - \mathbb{P}(X = -1)$
"""


def symmetric_binomial_inverse_cdf(p, x):

    if x <= 1-p:
        return -1.
    else:
        return 1.


def symmetric_binomial_expectation(p):
    return 2 * p - 1


def symmetric_binomial_variance(p):
    return 4 * p * (1 - p)


"""
Discrete Probability Distributions
"""

"""
returns the probability that the Poisson distribution takes value $k \in \mathbb{N}_0$.
"""


def poisson_pdf(_lambda, k):

    p = np.exp(-_lambda) * np.power(_lambda, k) / math.factorial(k)
    return p


"""
returns the probability that the Binomial distribution $B(n,p)$ distribution takes value $k \in \{0,\ldots, n\}$.
"""


def binomial_pdf(p, k, n):

    # probability calculated "by hand" - is available in numpy 'off-the-shelf'

    p = np.power(1-p, n-k) * np.power(p, k) * math.factorial(n) / math.factorial(k) / math.factorial(n-k)
    return p
