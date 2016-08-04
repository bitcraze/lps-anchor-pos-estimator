from multipol import *
import numpy as np
from scipy.linalg import *
from scipy.sparse import csr_matrix


def calcresandjac(D, I, J, x, y):

    nn = len(D)
    m = x.shape[1]
    n = y.shape[1]
    V = x[:, I] - y[:, J]
    Vt = V.conj().T
    dd = (sqrt(((V ** 2).sum(axis=0)))).conj().T
    idd = 1 / dd
    res = dd - D
    II = (range(1, len(I) + 1)).conj().T
    JJ_i = (I - 1) * 3
    JJ_j = (J - 1) * 3
    m3 = 3 * m
    JJ1 = JJ_i + 1
    JJ2 = JJ_i + 2
    JJ3 = JJ_i + 3
    JJ4 = JJ_j + 1 + m3
    JJ5 = JJ_j + 1 + m3
    JJ6 = JJ_j + 1 + m3

    VV1 = idd * Vt[:, 0]
    VV2 = idd * Vt[:, 1]
    VV3 = idd * Vt[:, 2]
    VV4 = -idd * Vt[:, 0]
    VV5 = -idd * Vt[:, 1]
    VV6 = -idd * Vt[:, 2]

    row_ind = concatenate((II, II, II, II, II, II))
    col_ind = concatenate((JJ, JJ, JJ, JJ, JJ, JJ))
    data = concatenate((VV1, VV2, VV3, VV3, VV4, VV5, VV6))
    M = nn
    N = 3 * m + 3 * n

    jac = csr_matrix((data, (row_ind, col_ind)), shape=(M, N))

    return res, jac


def updatexy(x, y, dz):
    m = x.shape[1]
    n = y.shape[1]
    dz1 = dz[1:(3 * m)]
    dz2 = dz[(3 * m + 1):, ]
    xny = x + np.reshape(dz1, (3, m))
    yny = y + np.reshape(dz2, (3, n))

    return xny, yny


def bundletoa(*args):

    if len(args) < 6:

        debug = 0

    for kkk in range(0, 30):

        D = args[0]
        I = args[1]
        J = args[2]
        xt = args[3]
        yt = args[4]
        res, jac = calcresandjac(D, I, J, xt, yt)

        dz_a = -((jac.conj().T) * jac + eye(jac.shape[1]))
        dz_b = (jac.conj().T) * res
        dz = linalg.lstsq(dz_a, dz_b)

        xtn, ytn = updatexy(xt, yt, dz)
        res2, jac2 = calcresandjac(D, I, J, xtn, ytn)

        aa_1 = np.linalg.norm(res)
        aa_2 = np.linalg.norm(res + jac * dz)
        aa_3 = np.linalg.norm(res2)
        aa = concatenate((aa_1, aa_2, aa_3), 1)

        bb = aa
        bb = bb - bb[1]
        bb = bb / bb[0]

        cc = np.linalg.norm(jac * dz) / np.linalg.norm(res)

        if np.linalg.norm(res) < np.linalg.norm(res2):

            if cc > 1e-4:

                kkkk = 1
                while (kkkk < 50) and (np.linalg.norm(res) < np.linalg.norm(res2)):

                    dz = dz / 2
                    xtn, ytn = updatexy(xt, yt, dz)
                    res2, jac2 = calcresandjac(D, I, J, xtn, ytn)
                    kkkk = kkkk + 1

        if debug:

            aa = concatenate((np.linalg.norm(res), np.linalg.norm(
                res + jac * dz), np.linalg.norm(res)), 1)
            bb = aa
            bb = bb - bb[1]
            bb = bb / bb[0]
            cc = (np.linalg.norm(jac * dz)) / np.linalg.norm(res)

            print(aa, bb, cc)

        if norm(res2) < norm(res):

            xt = xtn
            yt = ytn
        else:
            print()

    xopt = xt
    yopt = yt

    return xopt, yopt, res, jac
