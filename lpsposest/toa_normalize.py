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
import numpy as np
import scipy
from numpy.core.umath import sign


def toa_normalize(x0, y0):
    xdim = x0.shape[0]
    m = x0.shape[1]
    n = x0.shape[1]

    t = -x0[:, 1]
    x = x0 + np.tile(t, (1, m))
    y = y0 + np.tile(t, (1, n))

    qr_a = x[:, 2:(1 + xdim)]
    q, r = scipy.linalg.qr(qr_a)

    x = (q.conj().T) * x
    y = (q.conj().T) * y
    M = np.diag(sign(np.diag(qr_a)))
    x1 = M * x
    y1 = M * y

    return x1, y1
