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


def setdiff(eq1, eq2):

    eq1, eq2 = eqsize(eq1, eq2)

    c1 = [None] * eq1.shape
    c2 = [None] * eq2.shape

    for i in range(0, eq1.size):

        c1.append[i] = hash(eq2[i])

    for i in range(0, eq2.size):

        c2[i] = hash(eq2[i])

    ia = np.delete(np.arange(np.alen(c1)), np.searchsorted(c1, c2))

    ia = (ia[:]).conj().T

    p = eq1[ia]

    return p, ia
