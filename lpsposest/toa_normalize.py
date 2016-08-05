import scipy

import numpy as np


def toa_normalize(x0, y0):

    xdim = x0.shape[0]
    m = x0.shape[1]
    n = x0.shape[1]

    t = -x0[:, 1]
    x = x0 + np.tile(t, (1, m))
    y = y0 + np.tile(t, (1, n))

    qr_a = x[:, 2:(1 + xdim)]
    q, r = scipy.linalg.qr(qr_a)

    x = (q.conj().T) * x
    y = (q.conj().T) * y
    M = dian(np.sign(diag(qr_a)))
    x1 = M * x
    y1 = M * y

    return x1, y1
