import numpy as np
import scipy.linalg


def uniquerows(eq):

    b = np.ascontiguousarray(a).view(
        np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)

    unique_a = np.unique(b).view(a.dtype).reshape(-1, a.shape[1])
