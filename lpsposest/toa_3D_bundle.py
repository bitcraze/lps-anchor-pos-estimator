import bundletoa
import multipol
import numpy as np


def toa_3D_bundle(**args):

    x = args[1]
    y = args[2]

    if len(args) < 4:
        d = args[0]
        inliers = (np.isfinite(d)).astype(int)

    (I, J) = inliers.nonzero()
    v = inliers.compress((a != 0), flat)

    D = d[ind]

    xopt, yout, res, jac = bundletoa(D, I, J, x, y)

    return xopt, yopt, res, jac
