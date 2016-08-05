import numpy as np
from scipy.sparse import csr_matrix


def calcresandjac(mdata, param):
    auxv = param.U[:, mdata.I22 - 1] * param.V[:, mdata.j22 - 1]
    Bhatv = np.concatenate((param.R, auxv.conj().T))
    d2vc = mdata.bhat2d2 * Bhatv
    res = d2vc - mdata.d2vm

    II1 = (param.indzr).conj().T
    JJ1 = II1
    dBdz = np.ones(II1.shape)

    auxv1 = range((len(param.R) + 1), (len(param.R) + len(mdata.I22)))
    II2 = auxv1.conj().T

    dBdU1 = (param.V[0, mdata.J22 - 1]).conj().T
    dBdU2 = (param.V[1, mdata.J22 - 1]).conj().T
    dBdU3 = (param.V[2, mdata.J22 - 1]).conj().T
    dBdV1 = (param.U[0, mdata.J22 - 1]).conj().T
    dBdV2 = (param.U[1, mdata.J22 - 1]).conj().T
    dBdV3 = (param.U[2, mdata.J22 - 1]).conj().T
    JJ21 = (mdata.I22 - 1) * 3 + 1 + param.nzr
    JJ22 = (mdata.I22 - 1) * 3 + 2 + param.nzr
    JJ23 = (mdata.I22 - 1) * 3 + 3 + param.nzr
    JJ24 = (mdata.J22 - 1) * 3 + 1 + 3 * (param.mm - 1) + param.nzr
    JJ25 = (mdata.J22 - 1) * 3 + 2 + 3 * (param.mm - 1) + param.nzr
    JJ26 = (mdata.J22 - 1) * 3 + 3 + 3 * (param.mm - 1) + param.nzr

    row_ind = np.concatenate((II1, II2, II2, II2, II2, II2, II2))
    col_ind = np.concatenate((JJ1, JJ21, JJ22, JJ23, JJ24, JJ25, JJ26))
    data = np.concatenate((dBdz, dBdU1, dBdU2, dBdU3, dBdV1, dBdV2, dBdV3))
    M = len(Bhatv)
    N = 3 * (param.mm - 1) + 3 * (param.nn - 1) + param.nzr

    jac = csr_matrix((data, (row_ind, col_ind)), shape=(M, N)).toarray()

    jac[:, (param.nzr + 1):(param.nzr + 9)] = []

    jac = mdata.bhat2d2 * jac

    return res, jac
