from math import sqrt

import numpy as np
from multipol import Multipol
from polynomials2matrix import polynomials2matrix
from scipy import interpolate
from tm_bundle_rank import tm_bundle_rank
from tm_ransac5rows import tm_ransac5rows
from tm_ransac_more_cols import tm_ransac_more_cols
from tm_ransac_more_rows import tm_ransac_more_rows
from toa_3D_bundle import toa_3D_bundle
from toa_calc_d_from_xy import toa_calc_d_from_xy
from toa_normalize import toa_normalize


def system_misstoa_ransac_bundle(d, sys):
    dobundle = 1

    sol = None

    for kk in range(1, 2):

        sol, manrin = tm_ransac5rows(d, sys)

        if dobundle:
            sol, res0, res, d2calc = tm_bundle_rank(sol, d)

        sol = tm_ransac_more_rows(d, sol, sys)

        if dobundle:
            sol, res0, res, d2calc = tm_bundle_rank(sol, d)

        sol = tm_ransac_more_cols(d, sol, sys)

        if dobundle:
            sol, res0, res, d2calc = tm_bundle_rank(sol, d)

        sol = tm_ransac_more_rows(d, sol, sys)

        if dobundle:
            sol, res0, res, d2calc = tm_bundle_rank(sol, d)

        sol = tm_ransac_more_cols(d, sol, sys)

        if dobundle:
            sol, res0, res, d2calc = tm_bundle_rank(sol, d)

    Bhat = sol.Bhat
    D = 3
    u, s, v = np.svd(Bhat[2:, 2:])
    xr = u[:, 0:D - 1]
    yr = (s[0:D - 1, 0:D - 1]) * (v[:, 0:D - 1])
    auxvar1 = np.zeros((D, 1))
    xtp = np.concatenate((auxvar1, xr))
    yt = np.concatenate((auxvar1, yr))
    xt = xtp / (-2)  # maybe np.divide(xtp,-2)
    Bhatcol1 = Bhat[:, 0]

    nr_of_unknowns = (D * (D + 1)) / 2 + D + 1

    xv = [None] * nr_of_unknowns
    for i in range(0, nr_of_unknowns):
        xv[i] = Multipol.multipol(1, np.zeros(nr_of_unknowns, 1))

    bv = None
    Cv = None

    if D == 3:
        Cv1 = np.concatenate((xv[1], xv[2], xv[3]), 1)
        Cv2 = np.concatenate((xv[2], xv[4], xv[5]), 1)
        Cv3 = np.concatenate((xv[3], xv[5], xv[6]), 1)
        Cv = np.concatenate((Cv1, Cv2, Cv3))

        bv = np.concatenate((xv[6], xv[7], xv[9]), 1)
        bv = bv.conj().T
    elif D == 2:
        Cv1 = np.concatenate((xv[1], xv[2]), 1)
        Cv2 = np.concatenate((xv[2], xv[3]), 1)
        Cv = np.concatenate((Cv1, Cv2))

        bv = np.concatenate((xv[3], xv[5]), 1)
        bv = bv.conj().T

    eqs = [None] * xt.shape[1]
    for i in range(1, xt.shape[1]):
        eqs[i - 1] = (-2 * (xt[:, i]).conj().T * bv +
                      (xt[:, i]).conj().T * Cv * xt[:, i]) - Bhatcol1[i]

    eqs_linear = eqs
    cfm_linear, mons_linear = polynomials2matrix(eqs_linear)
    cfm_linear = np.asarray(cfm_linear)
    cfm_linear = cfm_linear / np.tile(sqrt(sum(cfm_linear ** 2, 2)),
                                      (1, cfm_linear.shape[1]))
    cfm_linear0 = cfm_linear
    AA = cfm_linear0[:, 1:-2]
    bb = cfm_linear0[:, -1]
    zz0 = -np.linalg.pinv(AA) * bb

    H = interpolate.interp1d(Cv, (np.append(zz0, 0)).reshape(10, 1))
    b = interpolate.interp1d(bv, (np.append(zz0, 0,)).reshape(10, 1))

    if min(np.linalg.eig(H)):
        L = np.linalg.cholesky(np.linalg.inv(H)).T
    else:
        mins = min(np.linalg.eig(H))
        H = H + (-mins + 0.1) * np.eye(3)
        L = np.linalg.cholesky(np.linalg.inv(H)).T

    r00 = np.linalg.inv(L.conj().T) * xt
    s00 = L * (yt + np.tile(b, (1, Bhat.shape[1])))

    r0 = np.zeros(3, d.size[0])
    s0 = np.zeros(3, d.size[1])

    r0[:, sol.rows] = r00
    s0[:, sol.cols] = s00

    toa_calc_d_from_xy(r0, s0)

    inliers = sol.inlmatrix == 1

    r1, s1, res, jec = toa_3D_bundle(d, r0, s0, inliers)
    r, s = toa_normalize(r1, s1)

    return r, s, inliers


print(__name__)
