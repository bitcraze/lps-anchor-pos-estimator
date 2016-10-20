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


def issym(x):
    return 'sympy.' in str(type(x))


class Multipol:
    coeffs = 0
    monomials = 0

    def __init__(self, coeffs, monomials):
        self.coeffs = coeffs
        self.monomials = monomials

    @staticmethod
    def multipol(*args):
        if (type(args[0]) is int or type(
                args[0]) is list or type(
            args[0]) is np.ndarray) and \
                (type(args[1]) is int or type(args[1]) is list or type(
                    args[1]) is np.ndarray):
            coeffs = args[0]
            monomials = args[1]
            if not coeffs or coeffs.size == 0:
                coeffs = 0
            if not monomials or monomials.size == 0:
                monomials = 0
            if monomials.size != coeffs.size:
                assert False
            s = Multipol(coeffs, monomials)

        else:
            assert False

        return s


print(__name__)
