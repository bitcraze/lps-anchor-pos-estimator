import numpy as np


def hash(p):

    m = (abs(p.coeffs)).max(0)
    A = np.concatenate((p.coeffs / m * 65535, p.monomials))

    B = np.concatenate((abs(A), np.sign(A) + 1))
    h = chr(np.concatenate((np.log(m + 1), B[:])))

    return h
