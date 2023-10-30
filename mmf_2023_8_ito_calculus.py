"""
Author: Thomas Balzer
(c) 2023
Material for MMF Stochastic Analysis - Fall 2023
"""


import numpy as np
import plot_utilities as pu

"""
-- Generic Base Class of C^{1,2} functions with derivatives where Ito's formula applies
"""


class FunctionC12:

    def __init__(self, funct):
        self.funct = funct
        self.dft = 0.

    def definition(self):
        return str('Not implemented')

    def value(self, _t, _x):
        return self.funct.value(_x)

    def f_t(self, _t, _x):
        return self.dft

    def f_x(self, _t, _x):
        return self.funct.f_x(_x)

    def f_xx(self, _t, _x):
        return self.funct.f_xx(_x)


class Function1D:

    def value(self, _x):
        return 0.

    def f_x(self, _x):
        return 0.

    def f_xx(self, _x):
        return 0.


"""
$$F(t,x) = x^{\alpha}$$
"""


class GenericMonomial(Function1D):

    def __init__(self, _alpha):
        self.alpha = _alpha

    def definition(self):
        return str('$x^{\alpha}$ for alpha = ' + str(self.alpha))

    def value(self, _x):
        return np.power(_x, self.alpha)

    def f_x(self, _x):
        return self.alpha * np.power(_x, self.alpha - 1)

    def f_xx(self, _x):

        return 0. if self.alpha == 1. else self.alpha * (self.alpha - 1) * np.power(_x, self.alpha - 2)


"""
-- F(t,x) = \exp(x)
"""


class ExponentialFunction(Function1D):

    def value(self, _x):
        return np.exp(_x)

    def definition(self):
        return str('$\exp(x)$')

    def f_x(self, _x):
        return self.value(_x)

    def f_xx(self, _x):
        return self.value(_x)


"""
-- F(t,x) = \ln(1. + t) * x
"""


class LogTX(FunctionC12):

    def __init__(self):
        pass

    def definition(self):
        return str('$\ln(1+t) x$')

    def value(self, _t, _x):
        return np.log(1. + _t) * _x

    def f_t(self, _t, _x):
        return _x / (1. + _t)

    def f_x(self, _t, _x):
        return np.log(1 + _t)

    def f_xx(self, _t, _x):
        return 0.


"""
-- F(t,x) = \ln(1. + t) * x
"""


class VarB(FunctionC12):

    def __init__(self):
        pass

    def definition(self):
        return str('$t + x^2$')

    def value(self, _t, _x):
        return _t + _x * _x

    def f_t(self, _t, _x):
        return 1.

    def f_x(self, _t, _x):
        return 2. * _x

    def f_xx(self, _t, _x):
        return 2.


"""
-- F(t,x) = \ln(x)
"""


# class LogarithmicFunction(Function1D):
#
#     def value(self, _x):
#         return np.log(_x)
#
#     def f_x(self, _x):
#         return 1. / _x
#
#     def f_xx(self, _x):
#         return - 1. / (_x * _x)


class ExpTX(FunctionC12):

    def __init__(self):
        pass

    def definition(self):
        return str('$\exp(tx)$')

    def value(self, _t, _x):
        return np.exp(_t * _x)

    def f_t(self, _t, _x):
        return _x * self.value(_t, _x)

    def f_x(self, _t, _x):
        return _t * self.value(_t, _x)

    def f_xx(self, _t, _x):
        return _t * _t * self.value(_t, _x)


class TTimesX(FunctionC12):

    def __init__(self):
        pass

    def definition(self):
        return str('$t x$')

    def value(self, _t, _x):
        return _t * _x

    def f_t(self, _t, _x):
        return _x

    def f_x(self, _t, _x):
        return _t

    def f_xx(self, _t, _x):
        return 0.


def generate_brownian_path(steps, dt):

    path_incr = np.random.normal(0., np.sqrt(dt), steps)
    path = [sum(path_incr[:k]) for k in range(steps)]
    return path


def value_single_path(driving_process_path, time_grid, f_w):

    """
    $$ \int_0^T F_t(t, X(t)) dt
     + \int_0^T F_x(t, X(t)) dX(t)
     + \frac{1}{2} F_xx(t, X(t)) dt
    $$
    """

    evaluate_at_lhs = True
    value = 0.
    t_prev = time_grid[0]
    x_prev = driving_process_path[0]
    for k in range(1, len(time_grid)):
        t = time_grid[k]
        x = driving_process_path[k]
        dt = t - t_prev
        dx = x - x_prev
        x_v = (x_prev if evaluate_at_lhs else x)
        t_v = (t_prev if evaluate_at_lhs else t)
        value = (value + f_w.f_t(t_v, x_v) * dt
                 + f_w.f_x(t_v, x_v) * dx + 0.5 * f_w.f_xx(t_v, x_v) * dt)
        t_prev = t
        x_prev = x

    return value


if __name__ == '__main__':

    time = 1.
    time_steps = 250
    delta_t = time / time_steps
    grid = [k * delta_t for k in range(time_steps)]

    ito_values = []
    term_values = []
    fw = ExpTX()
    # fw = VarB()
    # fw = VarB()
    # fw = TTimesX()
    # fw = FunctionC12(GenericMonomial(1.))
    # fw = FunctionC12(ExponentialFunction())
    n_paths = 5000
    for p in range(n_paths):
        x_path = generate_brownian_path(time_steps, delta_t)
        ito_values.append(value_single_path(x_path, grid, fw))
        term_values.append(fw.value(time, x_path[time_steps-1]) - fw.value(0., 0.))

    mp = pu.PlotUtilities("Histogram of $F(T, B(T))$ for $F(t,x)=${0}".format(fw.definition()), 'Outcome', 'Rel. Occurrence')
    mp.plot_histogram([ito_values, term_values], 50, ['Integral Values', 'Exact Values'])
