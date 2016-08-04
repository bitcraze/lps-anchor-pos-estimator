import numpy as np
import scipy.linalg
from eqsize import *
from nterms import *
from multipol import *
from unique import *


def polynomials2matrix(*args):
    if len(args) < 2:
        order = 'grevlex'

    p = args[0]
    p = eqsize(p)
    nt = sum(nterms(p))
    nv = nvars(p[0])

    M = np.zeros((nv, nt), dtype=np.int32)
    inds = [none] * p.size
    k = 0
    for i in range(0, p.size):
        inds.append[i] = range(k, k + nterms(p[i]) - 1)
        M[:, k:k + nterms(p[i]) - 1] = monomials(p[i])
        k = k + nterms(p[i])

    if order == 'grevlex':
        neg_M_sum = -1 * M.sum(axis=0)
        M_trans = M.conj().T
        neg_M_sum_trans = neg_M_sum.conj().T
        M_fliplr = fliplor(M_trans)

        new_grev_M = np.concatnate(neg_M_sum_trans, M_fliplr)

        _, ia, ib = unique(new_grev_M)

    M = float(M[:, ia])

    for i in range(M.shape[1], -1, -1):
        mon[i, 1] = multipol(1, M[:, i])

    C = zeros(p.size, M.shape[0])
    for i in range(0, p.size):
        ind = ib[inds[i]]
        C[i, ind] = coeffs(p[i])

    return C, mon
