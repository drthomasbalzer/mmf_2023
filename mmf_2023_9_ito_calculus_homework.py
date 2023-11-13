"""
Author: Thomas Balzer
(c) 2023
Material for MMF Stochastic Analysis - Fall 2023
"""


import numpy as np
import plot_utilities as pu

from mmf_2023_8_ito_calculus import VarB, ExpTX, FunctionC12
from mmf_2023_8_ito_calculus import generate_brownian_path, value_single_path


class LN1PlusTTimesX(FunctionC12):

    def __init__(self):
        pass

    def definition(self):
        return str('$\ln(1+t)x$')

    def value(self, _t, _x):
        return np.log(1 + _t) * _x

    def f_t(self, _t, _x):
        return _x / (1 + _t)

    def f_x(self, _t, _x):
        return np.log(1 + _t)

    def f_xx(self, _t, _x):
        return 0.


if __name__ == '__main__':

    time = 1.
    time_steps = 200
    delta_t = time / time_steps
    print('Time step: ' + str(delta_t))
    time_grid = [k * delta_t for k in range(time_steps)]

    '''
    simulation of Brownian paths
    '''
    n_paths_ex = 10
    paths = []
    for k in range(n_paths_ex):
        p = generate_brownian_path(time_steps, delta_t)
        paths.append(p)

    mp = pu.PlotUtilities("Example Brownian Motion Paths", 'Time', 'Outcome')
    mp.multi_plot(time_grid, paths)

    f1 = VarB()

    n_paths = 5000
    x_paths = []
    for p in range(n_paths):
        x_paths.append(generate_brownian_path(time_steps, delta_t))

    term_values = []
    for p in range(n_paths):
        term_values.append(f1.value(time, x_paths[p][time_steps-1]) - f1.value(0., 0.))

    mp = pu.PlotUtilities("Histogram of $F(T, B(T))$ for $F(t,x)=${0}".format(f1.definition()),
                          'Outcome', 'Rel. Occurrence')
    mp.plot_histogram([term_values], 50, ['Exact Values'])

    # Evaluation of Ito Formula at LHS
    lhs_values = []
    for p in range(n_paths):
        lhs_values.append(value_single_path(x_paths[p], time_grid, f1, True))

    mp.plot_histogram([term_values, lhs_values], 50, ['Exact Values', 'LHS Values'])

    # Evaluation of Ito Formula at RHS
    rhs_values = []
    for p in range(n_paths):
        rhs_values.append(value_single_path(x_paths[p], time_grid, f1, False))

    mp.plot_histogram([term_values, lhs_values, rhs_values], 50, ['Exact Values',  'LHS Values',  'RHS Values'])

    # Evaluation of Ito Formula at the Left Hand Side for two other functions
    functions = [ExpTX(), LN1PlusTTimesX()]
    for f in functions:
        lhs_values = []
        term_values = []
        for p in range(n_paths):
            term_values.append(f.value(time, x_paths[p][time_steps-1]) - f.value(0., 0.))
            lhs_values.append(value_single_path(x_paths[p], time_grid, f, True))

        mp = pu.PlotUtilities("Histogram of $F(T, B(T))$ for $F(t,x)=${0}".format(f.definition()),
                              'Outcome', 'Rel. Occurrence')
        mp.plot_histogram([term_values, lhs_values], 50, ['Exact Values', 'LHS Values'])
