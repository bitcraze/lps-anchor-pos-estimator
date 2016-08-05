import numpy as np
from bundletoa import bundletoa


def toa_3D_bundle(d, x, y, inliers):
    (I, J) = inliers.nonzero()

    ind = np.ravel_multi_index((I, J), dims=d.shape)
    D = d[ind]

    xopt, yopt, res, jac = bundletoa(D, I, J, x, y)

    return xopt, yopt, res, jac
