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
from eqsize import eqsize


def unique(eq):
    eq = eqsize(eq)
    c1 = [None] * eq.shape
    for i in range(0, eq.size):
        c1.append[i] = hash(eq[i])

    c1 = np.asarray(c1)

    if c1.ndim == 1:
        _, ia, ic = np.unique(c1, return_index=True, return_inverse=True)
        ia = (ia[:, ]).conj().T
        ic = (ic[:, ]).conj().T
        u = eq[ia]

    else:
        a = c1
        b = np.ascontiguousarray(a).view(
            np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
        _, ia, ic = np.unique(b, return_index=True, return_inverse=True)

    return u, ia, ic
