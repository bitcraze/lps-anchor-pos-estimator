#
#    ||          ____  _ __
# +------+      / __ )(_) /_______________ _____  ___
# | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
# +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#  ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
#
# LPS Anchor Position Estimator
#
# Copyright (C) 2016 Bitcraze AB
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,USA
#
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
