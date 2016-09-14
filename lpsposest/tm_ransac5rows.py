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
            tmp21 = tmp2[0:4]
            tmp22 = tmp2[4:, ]

            cl,_ = compactionmatrix(5)

            cr, _ = compactionmatrix(tmp2.shape[0])
            Btmp1 = np.dot(cl, B[:, tmp2])
            Btmp = np.dot(Btmp1,cr.conj().T)

            B1 = Btmp[:, 0:3]
            B2 = Btmp[:, 3:]

            u, s, v = linalg.svd(B1)
            u4 = u[:, 3]

            if 0:
                abs((u4.conj().T) * B2)

            Imiss = isnan(d)
            auxvar3 = abs((u4.conj().T) * B2)
            okind = (auxvar3 > sys.ransac_threshold).nonzero()
            inlim = zeros(d.shape)
            inlim = inlim - Imiss
            auxvar4 = concatenate((tmp21, tmp22[okind]), 1)
            inlim[tmprows, okcol[auxvar4]] = ones((5, 4 + okind.size))
            nrinl = 4 + okind.size

            if nrinl > maxnrinl:
                maxnrinl = nrinl

                sol.rows = tmprows
                sol.cols = okcol[(concatenate((tmp21, tmp22[okind]), 1))]
                sol.row1 = sol.rows[1]
                sol.col1 = sol.cols[1]
                sol.inlmatrix = inlim
                B = d2[sol.rows, sol.cols]
                cl, dl = compactionmatrix(B.shape[0])
                cr, dr = compactionmatrix(B.shape[1])
                Bhat = dl * B * dr.conj().T
                Btilde = cl * B * cr.conj().T
                u, s, vh = linalg.svd(Btilde)
                v = vh.T
                s[3:, ] = zeros(s.shape[0] - 3, s.shape[1])
                Btilde = u * s * (v.conj().T)
                Bhat[1:, 1:] = Btilde
                sol.Bhat = Bhat
                sol.dl = dl
                sol.dr = dr

    return sol, maxnrinl
