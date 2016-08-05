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
