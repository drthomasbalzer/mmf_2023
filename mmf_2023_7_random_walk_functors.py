"""
Author: Thomas Balzer
(c) 2023
Material for MMF Stochastic Analysis - Fall 2023
"""

"""
Multi-Period Maximising Probability to Reach a Goal
"""

import plot_utilities as pu
import numpy as np

class Functor:

    def value(self, x):

        return 0.

class IdentityFunction(Functor):

    def value(self, x):

        return x


class ExcessProbability(Functor):

    def __init__(self, goal):

        self.goal = goal

    def value(self, x):

        return 1. if x > self.goal else 0.


class CallOptionPayoff(Functor):

    def __init__(self, strike):

        self.strike = strike

    def value(self, x):

        return x - self.strike if x > self.strike else 0.


class RandomWalk:

    def __init__(self, p):

        self.p = p

class RandomWalkFunctor:

    def __init__(self, random_walk, base_function, scale = 1.):

        self.rw = random_walk
        self.f = base_function
        self.scale = scale

    def value(self, x):

        return self.rw.p * self.f.value(x + 1. / self.scale) + (1 - self.rw.p) * self.f.value(x - 1./self.scale)

if __name__ == '__main__':

    print('Random Walk - Markov Transition Functor')

    _p = 0.5
    rw = RandomWalk(_p)
    # f = IdentityFunction()
    # f = ExcessProbability(0.)
    f = CallOptionPayoff(0.)
    lb = 10
    absc = [-lb + k for k in range(int(lb*2))]

    y_values = [f.value(a) for a in absc]
    mp = pu.PlotUtilities('Random Walk', 'x', 'y')
    mp.multi_plot(absc, [y_values])

    h_f = RandomWalkFunctor(rw, f)
    mp = pu.PlotUtilities('Random Walk - 1 Step Functor', 'x', 'y')
    mp.multi_plot(absc, [y_values, [h_f.value(a) for a in absc]])

    n = 20
    bf = f
    for m in range(1, n+1):
        # bf = RandomWalkFunctor(rw, bf, float(np.sqrt(n)))
        bf = RandomWalkFunctor(rw, bf)

    mp = pu.PlotUtilities('Random Walk - N-Step Functor', 'x', 'y')
    mp.multi_plot(absc, [y_values, [bf.value(a) for a in absc]])
    # mp.multi_plot([ float(a / n) for a in absc], [y_values, [bf.value(a) for a in absc]])
