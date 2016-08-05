import numpy as np


def nvars(p):
    n = np.zeros(p.shape)
    for i in range(p.size):
        n[i] = (p[i].monomials).shape[0]

    return n
