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
from compactionmatrix import compactionmatrix
from numpy import concatenate
from numpy import linalg
from numpy import ones
from numpy import random
from numpy import zeros
from numpy.core.umath import isfinite
from numpy.core.umath import isnan


def tm_ransac5rows(d, sys):
    class solstruct():
        pass

    sol = solstruct()

    maxnrinl = 0

    for iii in range(0, sys.ransac_k):

        d2 = d ** 2
        inl = (isfinite(d2)).astype(int)
        r_c = d2.shape
        m = r_c[0]
        tmprows = random.permutation(m)
        tmprows = tmprows[0:5]
        auxvar1 = inl[tmprows, :]
        auxvar2 = ((np.all(auxvar1, axis=0)).astype(int)).T
        okcol = (np.flatnonzero(auxvar2)).T

        B = d2[np.ix_(tmprows, okcol)]

        ntmp = B.shape[1]
        tmp2 = random.permutation(ntmp)

        if ntmp > 5:
            tmp21tup = tmp2[0:4]
            tmp21 = np.reshape(tmp21tup, (1, -1))
            tmp22tup = tmp2[4:, ]
            tmp22 = np.reshape(tmp22tup, (1, -1))
            cl, _ = compactionmatrix(5)

            cr, _ = compactionmatrix(tmp2.shape[0])
            Btmp1 = np.dot(cl, B[:, tmp2])
            Btmp = np.dot(Btmp1, cr.conj().T)

            B1 = Btmp[:, 0:3]
            B2 = Btmp[:, 3:]

            u, s, v = linalg.svd(B1)
            u4tup = u[:, 3]
            u4 = np.reshape(u4tup, (-1, 1))

            if 0:
                abs((u4.conj().T) * B2)

            Imiss = isnan(d)
            auxvar3 = abs(np.dot((u4.conj().T), B2))
            okindtup = (auxvar3 > sys.ransac_threshold).nonzero()
            okindmat = np.asarray(okindtup)
            okind = np.reshape(okindmat, (1, -1))
            inlim = zeros(d.shape)
            inlim = inlim - Imiss
            tmpconcat = concatenate((tmp21, tmp22[0, okind - 1]), 1)
            tmprows = np.reshape(tmprows, (-1, 1))
            inlim[tmprows, okcol[tmpconcat]] = ones((5, 4 + okind.size))
            nrinl = 4 + okind.size

            if nrinl > maxnrinl:
                maxnrinl = nrinl

                sol.rows = tmprows
                concatmat = concatenate((tmp21, tmp22[0, okind - 1]), 1)
                sol.cols = okcol[(concatmat)]
                sol.row1 = sol.rows[1]
                sol.col1 = sol.cols[0, 0]
                sol.inlmatrix = inlim
                B = d2[sol.rows, sol.cols]
                cl, dl = compactionmatrix(B.shape[0])
                cr, dr = compactionmatrix(B.shape[1])
                Bhatdotprod = np.dot(dl, B)
                Bhat = np.dot(Bhatdotprod, dr.conj().T)
                Btildedotprod = np.dot(cl, B)
                Btilde = np.dot(Btildedotprod, cr.conj().T)
                u, s, vh = linalg.svd(Btilde)
                v = vh.T
                s[3:, ] = zeros(s.shape[0] - 3, s.shape[1])
                Btilde = u * s * (v.conj().T)
                Bhat[1:, 1:] = Btilde
                sol.Bhat = Bhat
                sol.dl = dl
                sol.dr = dr

    return sol, maxnrinl
