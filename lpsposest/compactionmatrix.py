import numpy as np


def compactionmatrix(n):
    mat1 = -ones(n - 1, 1)
    mat2 = eye(n - 1)
    cc = concatenate((mat1, mat2), 1)
    mat3 = zeros(1, n - 1)
    auxcat = concatenate((1, mat3), 1)
    dd = concatenate((auxcat, cc))

    return cc, dd
