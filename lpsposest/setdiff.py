import multipol
import numpy as np
import scipy.linalg
from eqsize import *


def setdiff(eq1, eq2):

    eq1, eq2 = eqsize(eq1, eq2)

    c1 = [None] * eq1.shape
    c2 = [None] * eq2.shape

    for i in range(0, eq1.size):

        c1.append[i] = hash(eq2[i])

    for i in range(0, eq2.size):

        c2[i] = hash(eq2[i])

    ia = np.delete(np.arange(np.alen(c1)), np.searchsorted(c1, c2))

    ia = (ia[:]).conj().T

    p = eq1[ia]

    return p, ia
