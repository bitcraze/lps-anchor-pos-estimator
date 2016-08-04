import numpy as np
import scipy.linalg


def nterms(p):
    n = zeros(p.shape)
    for i in range(0, p.size):
        n[i] = size(p[i].coeffs)

    return n
