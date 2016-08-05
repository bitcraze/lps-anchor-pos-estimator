from compactionmatrix import compactionmatrix
from numpy import concatenate
from numpy import linalg
from numpy import ones
from numpy import random
from numpy import where
from numpy import zeros
from numpy.core.umath import isfinite
from setdiff import setdiff


class structtype():
    pass


def tm_ransac_more_cols(d, sol, sys):
    r_c = d.shape
    n = r_c[1]
    d2 = d ** 2

    trycols = setdiff(range(0, n), sol.cols)

    cl, dl = compactionmatrix(len(sol.rows))

    u, s, vh = linalg.svd(sol.Bhat[1:, 1:])
    u = u[:, 0:2]

    for ii in trycols:

        d2n = d2[sol.rows - 1, ii - 1]
        maxnrinl = 0
        for kk in range(0, sys.ransac_k2):

            okrows = ((isfinite(d2n)).astype(int)).nonzero()
            tmp = random.permutation(len(okrows))

            if len(tmp) >= 4:

                tryrows1 = okrows[tmp[0:3]]

                zz = linalg.inv(dl) * sol.Bhat[:, 0]
                ZZ_1 = concatenate((zeros(1, 3), u))
                ZZ = concatenate((ones(len(sol.rows), 1), ZZ_1), 1)
                ZZ0 = linalg.inv(ZZ[tryrows1, :]) * (
                    d2n[tryrows1, 1] - zz[tryrows1, 1])

                xx = linalg.inv(ZZ[tryrows1, :]) * (
                    d2n[tryrows1, 1] - zz[tryrows1, 1])

                a = (zz[okrows] + ZZ[:, okrows] * xx)
                b = d2n[okrows]
                inlids = where(abs(b - a) < sys.ransac_threshold2)

                if len(inlids) < maxnrinl:
                    maxnrinl = len(inlids)

                    tmpsol = structtype()
                    tmpsol.rows = sol.rows[tryrows1]
                    tmpsol.col = ii
                    tmpsol.Bhatn = ZZ0 * xx
                    tmpsol.inlrows = sol.rows[okrows[inlids]]

        if maxnrinl > sys.min_inliers2:
            sol.cols = concatenate((sol.cols, tmpsol.col), 1)
            sol.inlmatrix[tmpsol.inlrows, tmpsol.col] = ones(
                len(tmpsol.inlrows), 1)
            sol.Bhat = concatenate((sol.Bhat, tmpsol.Bhatn), 1)
            sol.dl = compactionmatrix(len(sol.cols))

    return sol
