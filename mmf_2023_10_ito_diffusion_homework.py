"""
Author: Thomas Balzer
(c) 2023
Material for MMF Stochastic Analysis - Fall 2023
"""

import numpy as np
import plot_utilities as pu

from mmf_2023_8_ito_calculus import FunctionC12
from mmf_2023_8_ito_calculus import generate_brownian_path
from mmf_2023_9_sde_examples import ItoProcessGBM, ItoProcessBMDrift, generate_ito_path, value_single_ito_path


class LNXMinusRT(FunctionC12):

    def __init__(self, r):
        self.r = r

    def definition(self):
        return str('$\ln(x) - rt$')

    def value(self, _t, _x):
        return np.log(_x) - self.r * _t

    def f_t(self, _t, _x):
        return -self.r

    def f_x(self, _t, _x):
        return 1. / _x

    def f_xx(self, _t, _x):
        return - 1. / (_x * _x)


if __name__ == '__main__':

    time = 1.
    time_steps = 200
    delta_t = time / time_steps
    time_grid = [k * delta_t for k in range(time_steps)]

    _a = 0.1
    _b = 0.2
    _r = 0.05
    f = LNXMinusRT(_r)
    ito_p = ItoProcessGBM(_a, _b, 1.)
    ito_t = ItoProcessBMDrift(_a - _r - 0.5 * _b * _b, _b, 0.)
    '''
    Sample the raw Brownian Motion paths 
    '''
    n_ex_paths = 10
    x_paths = []
    i_paths = []
    for p in range(n_ex_paths):
        _path = generate_brownian_path(time_steps, delta_t)
        x_paths.append(_path)
        ito_path = generate_ito_path(_path, time_grid, ito_p)
        i_paths.append(ito_path)

    mp = pu.PlotUtilities("Example Brownian Motion Paths", 'Time', 'Outcome')
    mp.multi_plot(time_grid, x_paths)

    mp = pu.PlotUtilities("Example Ito Diffusion Paths", 'Time', 'Outcome')
    mp.multi_plot(time_grid, i_paths)

    ''' 
    Sample the full paths (for both Brownian Motion and the Ito Process
    '''
    n_paths = 5000
    x_paths = []
    i_paths = []
    i_t_paths = []
    for p in range(n_paths):
        _path = generate_brownian_path(time_steps, delta_t)
        x_paths.append(_path)
        ito_path = generate_ito_path(_path, time_grid, ito_p)
        i_paths.append(ito_path)
        ito_t_path = generate_ito_path(_path, time_grid, ito_t)
        i_t_paths.append(ito_t_path)

    term_values = []
    for p in range(n_paths):
        term_values.append(f.value(time, i_paths[p][time_steps-1]) - f.value(0., ito_p.initial_value))

    mp = pu.PlotUtilities("Histogram of $F(T, X(T))$ for $F(t,x)=${0}".format(f.definition()),
                          'Outcome', 'Rel. Occurrence')
    mp.plot_histogram([term_values], 50, ['Exact Values'])

    int_values = []
    for p in range(n_paths):
        int_values.append(value_single_ito_path(i_paths[p], time_grid, f, ito_p))

    mp = pu.PlotUtilities("Histogram of $F(T, X(T))$ for $F(t,x)=${0}".format(f.definition()),
                          'Outcome', 'Rel. Occurrence')
    mp.plot_histogram([term_values, int_values], 50, ['Exact Values', 'Integral Values'])

    term_t_values = []
    for p in range(n_paths):
        term_t_values.append(i_t_paths[p][time_steps-1] - i_t_paths[p][0])

    mp = pu.PlotUtilities("Histogram of $F(T, X(T))$ for $F(t,x)=${0}".format(f.definition()),
                          'Outcome', 'Rel. Occurrence')
    mp.plot_histogram([term_values, int_values, term_t_values], 50, ['Exact Values', 'Integral Values', 'Direct Ito'])
