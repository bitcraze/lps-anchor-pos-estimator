import numpy as np


def compactionmatrix(n):
    mat1 = -np.ones((n-1, 1))
    mat2 = np.eye(n-1)
    cc = np.concatenate((mat1, mat2), 1)
    mat3 = np.zeros((1, n-1))
    auxcat = np.concatenate((1, mat3), 1)
    dd = np.concatenate((auxcat, cc))

    return cc, dd
