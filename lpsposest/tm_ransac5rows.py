import multipol
import numpy as np
import scipy.linalg
from compactionmatrix import *


def tm_ransac5rows(*argin):

    class struct():
        pass
    sol = solstruct()

    if len(argin) < 2:
        sys.ransac_threshold = 0.01
        sys.ransac_k = 5000

    maxnrinl = 0

    for iii in range(0, sys.ransac_k):

        d2 = d ** 2
        inl = np.isfinite(d2)
        r_c = d2.shape
        m = r_c[0]
        n = r_c[1]
        tmprows = np.random.permutation(m)
        tmprows = tmprows[0:4]
        auxvar1 = inl[tmprows, ]
        auxvar2 = np.all(auxvar1)
        okcol = (np.flatnonzero(auxvar2)).T

        B = d2[np.ix_(tmprows, okcol)]

        ntmp = B.shape[1]
        tmp2 = np.random.permutation(ntmp)

        if ntmp > 5:
            tmp21 = tmp2[1:4]
            tmp22 = tmp2[5:, ]

            cl = compactionmatrix(5)
            cr1 = compactionmatrix(4)
            cr2 = compactionmatrix(tmp22.shape[1])
            cr = compactionmatrix(tmp2.shape[1])

            Btmp = cl * B[:, tmp2] * cr.conj().T

            B1 = Btmp[:, 0:2]
            B2 = Btmp[:, 3:]

            u, s, v = linalg.svd(B1)
            u4 = u[:, 3]

            if 0:
                abs((u4.conj().T) * B2)
                I[:, tmp2]

            Imiss = np.isnan(d)
            auxvar3 = abs((u4.conj().T) * B2)
            okind = (auxvar3 > sys.ransac_threshold).nonzero()
            inlim = zeros(d.shape)
            inlim = inlim - Imiss
            auxvar4 = concatenate((tmp21, tmp22[okind]), 1)
            inlime[tmprows, okcol[auxvar4]] = ones((5, 4 + okind.size))
            nrinl = 4 + okind.size

            if nrinl > maxnrinl:
                if 0:
                    auxvar5 = abs((u4.conj().T) * B2)
                    okind = (auxvar5 > sys.ransac_threshold).nonzero()
                    inlim = zeros(d.shape)
                    inlim = inlim - Imiss
                    auxvar6 = concatenate((tmp21, tmp22[okind]), 1)
                    inlime[tmprows, okcol[auxvar6]] = ones((5, 4 + okind.size))
                    inlimgt = zeros(d.shape)
                    inlimgt = inlimgt - Imiss
                    inlimgt = inlimgt + Iinl

                maxnrinl = nrinl

                sol.rows = tmprows
                sol.cols = okcol[(concatenate((tmp21, tmp22[okind]), 1))]
                sol.row1 = sol.rows[1]
                sol.col1 = sol.cols[1]
                sol.inlmatrix = inlim
                B = d2[sol.rows, sol.cols]
                cl, dl = compactionmatrix(b.shape[0])
                cr, dr = compactionmatrix(b.shape[1])
                Bhat = dl * B * dr.conj().T
                Btilde = cl * B * cr.conj().T
                u, s, vh = linalg.svd(Btilde)
                v = vh.T
                s[3:, ] = zeros(s.shape[0] - e, s.shape[1])
                Btilde = u * s * (v.conj().T)
                Bhat[1:, 1:] = Btilde
                sol.Bhat = Bhat
                sol.dl = dl
                sol.dr = dr

    return sol, maxnrinl
