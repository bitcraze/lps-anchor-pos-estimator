import numpy as np
from multipol import *
import scipy.linalg
from scipy.sparse import csr_matrix


def updatexy(x, y, dz):

    m = x.shape[1]
    n = y.shape[1]
    dz1 = dz[1:3 * m]
    dz2 = dz[3 * m + 1, ]
    xny = x + reshape(dz1, (3, m))
    yny = y + reshape(dz2, (3, n))

    return xny, yny


def calcresandjac_cc(cc, x, y, lambdaa):

    j1 = cc[:, 0]
    j2 = cc[:, 1]
    j3 = cc[:, 2]

    res1 = y[0, j1 - 1] + y[0, j3 - 1] - 2 * y[0, j2 - 1]
    res2 = y[1, j1 - 1] + y[1, j3 - 1] - 2 * y[1, j2 - 1]
    res3 = y[2, j1 - 1] + y[2, j3 - 1] - 2 * y[2, j2 - 1]

    nn = cc.shape
    m = x.shape[1]
    n = y.shape[1]

    res = concatenate((res1.conj().T, res2.conj().T, res3.conj().T))
    res = lambdaa * res
    IIx = (range(1, nn + 1)).conj().T
    IIy = (range(nn + 1, 2 * nn + 1)).conj().T
    IIz = (range(2 * nn + 1, 3 * nn)).conj().T

    JJ1x = (j1 - 1) * 3 + 1 + 3 * m
    JJ1y = (j1 - 1) * 3 + 2 + 3 * m
    JJ1z = (j1 - 1) * 3 + 3 + 3 * m

    JJ2x = (j2 - 1) * 3 + 1 + 3 * m
    JJ2y = (j2 - 1) * 3 + 2 + 3 * m
    JJ2z = (j2 - 1) * 3 + 3 + 3 * m

    JJ3x = (j3 - 1) * 3 + 1 + 3 * m
    JJ3y = (j3 - 1) * 3 + 2 + 3 * m
    JJ3z = (j3 - 1) * 3 + 3 + 3 * m

    VV = ones(JJ1x.shape)

    row_ind = concatenate((IIx, IIx, IIx, IIy, IIy, IIy, IIz, IIz, IIz))
    col_ind = concatenate((JJx, JJx, JJx, JJy, JJy, JJy, JJz, JJz, JJz))
    data = concatenate((VVx, VVx, VVx, VVy, VVy, VVy, VVz, VVz, VVz))
    M = 3 * nn
    N = 3 * m + 3 * n

    jac = csr_matrix((data, (row_ind, col_ind)), shape=(M, N))

    return res, jac


def calcresandjac_toa(D, I, J, x, y):

    nn = len(D)
    m = x.shape[1]
    n = y.shape[1]
    V = x[:, I] - y[:, J]
    Vt = V.conj().T
    dd = (sqrt((V ** 2).sum)).conj().T

    idd = 1 / dd
    res = dd - D
    II = (range(1, len(I) + 1)).conj().T
    JJ1 = (I - 1) * 3 + 1
    JJ2 = (I - 1) * 3 + 2
    JJ3 = (I - 1) * 3 + 3
    JJ4 = (J - 1) * 3 + 1 + 3 * m
    JJ5 = (J - 1) * 3 + 2 + 3 * m
    JJ6 = (J - 1) * 3 + 2 + 3 * m

    VV1 = idd * Vt[:, 0]
    VV2 = idd * Vt[:, 1]
    VV3 = idd * Vt[:, 2]
    VV4 = -idd * Vt[:, 0]
    VV5 = -idd * Vt[:, 1]
    VV6 = -idd * Vt[:, 2]

    row_ind = concatenate((II, II, II, II, II, II))
    col_ind = concatenate((JJ, JJ, JJ, JJ, JJ, JJ))
    data = concatenate((VV1, VV2, VV3, VV4, VV5, VV6))
    M = nn
    N = 3 * m + 3 * n

    jac = csr_matrix((data, (row_ind, col_ind)), shape=(M, N))

    return res, jac


def calcresandjac(D, I, J, x, y, opts):

    res1, jac1 = calcresandjac_toa(D, I, J, x, y)
    if opts.cc.size != 0:
        res2, jac2 = calcresandjac_cc(opts.cc, x, y, opts.lambdacc)

    else:

        res2 = []
        jac2 = []

    res3 = []
    jac3 = []

    res = concatenate((res1, res2, res3))
    jac = concatenate((jac1, jac2, jac3))

    return res, jac


def bundletoa(D, I, J, xt, yt, debug=0, opts=[]):

    debug = 1

    for kkk in range(0, 10):

        res, jac = calcresandjac(D, I, J, xt, yt, opts)

        dz = linalg.lstsq(-((jac.conj().T) * jac + 0.1 *
                            eye(jac.shape[1])), (jac.conj().T) * res)

        xtn, ytn = updatexy(xt, yt, dz)
        res2, jac2 = calcresandjac(D, I, J, xt, yt, opts)

        aa_1 = np.linalg.norm(res)
        aa_2 = np.linalg.norm(res + jac * dz)
        aa_3 = np.linalg.norm(res2)
        aa = concatenate((aa_1, aa_2, aa_3), 1)

        bb = aa
        bb = bb - bb[1]
        bb = bb / bb[0]

        cc = np.linalg.norm(jac * dz) / np.linalg.norm(res)

        kkkk = 1

        if np.linalg.norm(res) < np.linalg.norm(res2):

            if cc > 1e-4:

                kkkk = 1
                while (kkkk < 50) and (np.linalg.norm(res) < np.linalg.norm(res2)):

                    dz = dz / 2
                    xtn, ytn = updatexy(xt, yt, dz)
                    res2, jac2 = calcresandjac(D, I, J, xtn, ytn, opts)
                    kkkk = kkkk + 1

        if debug:

            aa_1 = np.linalg.norm(res)
            aa_2 = np.linalg.norm(res + jac * dz)
            aa_3 = np.linalg.norm(res2)
            aa = concatenate((aa_1, aa_2, aa_3), 1)

            bb = aa
            bb = bb - bb[1]
            bb = bb / bb[0]

            cc = np.linalg.norm(jac * dz) / np.linalg.norm(res)

            print(aa, bb, cc)

        if np.linalg.norm(res2) < np.linalg.norm(res):

            xt = xtn
            yt = ytn
        else:

            if debug:
                print(kkk, ' stalled')

    xopt = xt
    yopt = yt

    return xopt, yopt, res, jac


def toa_3D_bundle_with_smoother(d, x, y, inliers=0, opts=[]):

    (I, J) = inliers.nonzero()
    D = inliers.compress((inliers != 0).flat)

    ind = np.ravel_multi_index((I, J), dims=d.shape, order='F')
    D = d[ind]

    opts = args[4]
    xopt, yopt, res, jac = bundletoa(D, I, J, xt, yt, 0, opts)

    return xopt, yopt, res, jac
