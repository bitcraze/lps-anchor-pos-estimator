from math import sqrt

import numpy as np


def toa_calc_d_from_xy(x, y):
    dimx = x.shape
    x_dim = dimx[0]
    m = dimx[1]

    dimy = y.shape
    y_dim = dimy[0]
    n = dimy[1]

    if x_dim > y_dim:
        y[(y_dim + 1):x_dim, ] = np.zeros(x_dim - y_dim, n)
    elif y_dim > x_dim:
        x[(x_dim + 1):y_dim, ] = np.zeros(y_dim - x_dim, n)

    d = sqrt(
        ((np.kron(np.ones(1, n), x) - np.kron(y, np.ones(1, m))) ** 2).sum(
            axis=0))
    d = np.asarray(d)
    d = np.reshape(d, (m, n), order='F')

    return d
