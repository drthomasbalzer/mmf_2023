"""
Author: Thomas Balzer
(c) 2023
Material for MMF Stochastic Analysis - Fall 2023
"""

import plot_utilities as pu
from mmf_2023_8_ito_calculus import generate_brownian_path

'''
Generic Representation of an Ito Process of the form 
dX(t) = (A(t)X(t) + a(t)) dt + (D(t) X(t) + d(t)) dB(t)
'''


class ItoProcessGeneral:

    def drift(self, _t, _x):
        return 0.

    def diffusion(self, _t, _x):
        return 0.

    def type(self):
        return r'Undefined'


class ItoProcessTH(ItoProcessGeneral):

    def __init__(self, _cap_a, _a, _cap_d, _d, _x):

        self._A = _cap_a
        self._a = _a
        self._D = _cap_d
        self._d = _d

        self.initial_value = _x

    def drift(self, _t, _x):
        return self._A * _x + self._a

    def diffusion(self, _t, _x):
        return self._D * _x + self._d

    def type(self):
        return r'General Time-Homogeneous'


class ItoProcessBMDrift(ItoProcessTH):

    def __init__(self, _a, _d, _x):

        self._A = 0.
        self._a = _a
        self._D = 0.
        self._d = _d
        self.initial_value = _x

    def type(self):
        return r'Brownian Motion With Drift'


class ItoProcessGBM(ItoProcessTH):

    def __init__(self, _cap_a, _cap_d, _x):

        self._A = _cap_a
        self._a = 0.
        self._D = _cap_d
        self._d = 0.
        self.initial_value = _x

    def type(self):
        return r'Geometric Brownian Motion'


class ItoProcessBrownianBridge(ItoProcessGeneral):

    def __init__(self, _cap_t, _b, _sigma, _x):

        self._T = _cap_t
        self._b = _b
        self._sigma = _sigma
        self.initial_value = _x

    def drift(self, _t, _x):
        return (self._b - _x) / (self._T - _t)

    def diffusion(self, _t, _x):
        return self._sigma

    def type(self):
        return r'Brownian Bridge'


def generate_ito_path(brownian_path, time_grid, _ito_process):

    process = _ito_process.initial_value
    path = [process]
    prev_bm_value = brownian_path[0]
    prev_t = time_grid[0]
    for bp, t in zip(brownian_path[1:], time_grid[1:]):
        # the paths will be constructed through application of Ito's formula
        _x = process
        _u = t
        this_bm_value = bp
        _du = t - prev_t
        _dBu = this_bm_value - prev_bm_value
        _a = _ito_process.drift(_u, _x)
        _b = _ito_process.diffusion(_u, _x)
        prev_bm_value = this_bm_value
        prev_t = t
        '''
        -- underlying ito process
        X(t + dt) = X(t) + a(t,x) dt + b(t,x) dB(t)
        '''
        process = process + _a * _du + _b * _dBu
        path.append(process)

    return path

def value_single_ito_path(driving_process_path, time_grid, f_w, _ito_process):

    """
    $$ \int_0^T F_t(t, X(t)) dt
     + \int_0^T F_x(t, X(t)) dX(t)
     + \frac{1}{2} F_xx(t, X(t)) dX(t)^2
    $$
    """

    value = 0.
    t_prev = time_grid[0]
    x_prev = driving_process_path[0]
    for k in range(1, len(time_grid)):
        t = time_grid[k]
        x = driving_process_path[k]
        dt = t - t_prev
        dx = x - x_prev
        x_v = x_prev
        t_v = t_prev
        _b = _ito_process.diffusion(t_v, x_v)
        value = (value + f_w.f_t(t_v, x_v) * dt
                 + f_w.f_x(t_v, x_v) * dx + 0.5 * f_w.f_xx(t_v, x_v) * _b * _b * dt)
        t_prev = t
        x_prev = x

    return value



def plot_sde(_max_time, _delta_t, _number_paths, _ito_process):

    # normals for a single paths
    size = int(_max_time / _delta_t)
    time_axis = [_delta_t * k for k in range(size)]

    paths = []
    for k in range(_number_paths):
        this_bm_path = generate_brownian_path(size, _delta_t)
        path = generate_ito_path(this_bm_path, time_axis, _ito_process)
        paths.append(path)

    # prepare and show plot
    mp = pu.PlotUtilities('Paths of Ito Process of Type {0}'.format(_ito_process.type()),
                          'Time', 'Value')
    mp.multi_plot(time_axis, paths)


if __name__ == '__main__':

    max_time = 5
    timestep = 0.001
    n_paths = 5

    '''
    Ito process of the form dX = (A X(t) + a) dt + (B X(t) + b) dB(t)
    '''

    # ito process of the form dX = a dt + b dB(t)
    ito_bm = ItoProcessBMDrift(.40, .2, 0)

    # ito process of the form dX = X(a dt + b dB(t))
    ito_exp = ItoProcessGBM(0., 0.2, 1)

    # ito process of the form dX = X(t) dt + b dB(t)
    ito_1 = ItoProcessTH(0.3, 0.0, 0., 0.5, 1)

    # ito process of the form dX = mean X(t) dt + b dB(t)
    ito_2 = ItoProcessTH(-0.05, 0.0, 0., 0.05, 1)

    # ito process of the form dX = mean X(t) dt + b dB(t)
    ito_mr = ItoProcessTH(-.5, 1, 0., 0.5, 10)

    # Brownian Bridge
    ito_bb = ItoProcessBrownianBridge(max_time, 1., .75, 0.)

    ito = ito_mr

    plot_sde(max_time, timestep, n_paths, ito)
