from numpy import size
from numpy import zeros


def nterms(p):
    n = zeros(p.shape)
    for i in range(0, p.size):
        n[i] = size(p[i].coeffs)

    return n
