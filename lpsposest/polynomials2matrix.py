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
from coeffs import coeffs
from eqsize import eqsize
from monomials import monomials
from multipol import Multipol
from nterms import nterms
from numpy import concatenate
from numpy import fliplr
from numpy import int32
from numpy import zeros
from nvars import nvars
from unique import unique


def polynomials2matrix(polynomial):
    p = eqsize(polynomial)
    nt = sum(nterms(p))
    nv = nvars(p[0])

    M = zeros((nv, nt), dtype=int32)
    inds = [None] * p.size
    k = 0
    for i in range(0, p.size):
        inds.append[i] = range(k, k + nterms(p[i]) - 1)
        M[:, k:k + nterms(p[i]) - 1] = monomials(p[i])
        k = k + nterms(p[i])

    neg_M_sum = -1 * M.sum(axis=0)
    M_trans = M.conj().T
    neg_M_sum_trans = neg_M_sum.conj().T
    M_fliplr = fliplr(M_trans)

    new_grev_M = concatenate(neg_M_sum_trans, M_fliplr)

    _, ia, ib = unique(new_grev_M)

    M = float(M[:, ia])

    mon = zeros(M.shape[1], 1)
    for i in range(M.shape[1], -1, -1):
        mon[i, 1] = Multipol.multipol(1, M[:, i])

    C = zeros(p.size, M.shape[0])
    for i in range(0, p.size):
        ind = ib[inds[i]]
        C[i, ind] = coeffs(p[i])

    return C, mon
