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


def eqsize(*args):
    m = 0
    varargout = [None] * (len(args) + 1)

    for a in range(0, len(args)):
        p1 = args[a]

        for i in range(0, p1.size):
            m = np.maximum(m, (p1[i].monomials).shape[0])

    for a in range(len(args), -1, -1):
        p1 = args[a]
        for i in range(0, p1.size):
            if (p1[i].monomials).shape[0] < m:
                p1[i].monomials[m, :] = 0

        varargout[a] = p1

    return varargout


print(__name__)
